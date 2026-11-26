"""Tests for the per pixel ray gradients returned by the eval3d backward pass.

The 3DGEER backward kernel already differentiates the Gaussian response with
respect to the ray that it generated from the camera model.  These tests pin
down that the world space ray gradients are now accumulated per pixel and
handed back through ``rasterize_to_pixels_eval3d``.

Three independent checks are used:

* a translation identity, which ties ``v_rays_o`` to ``v_means`` exactly,
* a pure PyTorch reimplementation of the eval3d compositing, differentiated by
  autograd and compared per pixel against the kernel,
* a parity check that asking for ray gradients leaves every other gradient
  untouched.
"""

import math
import sys

import pytest
import torch

device = torch.device("cuda:0")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")


W, H = 64, 48
FX = FY = 42.0
CX, CY = W / 2.0, H / 2.0
TILE_SIZE = 16
MIN_ALPHA = 1.0 / 255.0
MIN_T = 1e-4
MIN_2D_NORM = 1e-6


def _Ks():
    return torch.tensor(
        [[[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )


def _viewmat():
    """A camera that is both rotated and translated, so the test is not trivial."""
    angle = 0.3
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = torch.tensor(
        [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]],
        device=device,
        dtype=torch.float32,
    )
    translation = torch.tensor([0.2, -0.1, 0.05], device=device, dtype=torch.float32)
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = translation
    return viewmat[None]


def _radial_coeffs(camera_model):
    if camera_model == "fisheye":
        # All zeros is the equidistant Kannala Brandt model.
        return torch.zeros((1, 4), device=device, dtype=torch.float32)
    return None


def scene(num_gaussians=40, seed=0):
    """A scene with moderate opacities, so no pixel saturates the alpha clamp."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.float32).to(device)

    # Gaussians live in front of the camera, which itself sits near the origin.
    offset = torch.tensor([0.0, 0.0, 2.6], device=device)
    extent = torch.tensor([1.6, 1.2, 1.6], device=device)
    means = (rand(num_gaussians, 3) - 0.5) * extent + offset
    quats = torch.nn.functional.normalize(rand(num_gaussians, 4) - 0.5, dim=-1)
    # Well above the pixel footprint at this depth, so nothing degenerates.
    scales = 0.10 + 0.14 * rand(num_gaussians, 3)
    opacities = 0.25 + 0.25 * rand(num_gaussians)
    colors = rand(num_gaussians, 3)
    return means, quats, scales, opacities, colors


def pixel_rays(camera_model, viewmat):
    """World space rays through the pixel centres, as the kernel builds them."""
    rows = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    cols = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    py, px = torch.meshgrid(rows, cols, indexing="ij")
    u = (px - CX) / FX
    v = (py - CY) / FY

    if camera_model == "pinhole":
        rays_cam = torch.stack((u, v, torch.ones_like(u)), dim=-1)
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
    else:
        theta = torch.sqrt(u * u + v * v)
        # At the principal point the camera model returns [0, 0, 1] instead of
        # dividing by a vanishing radius.
        scale = torch.where(
            theta >= MIN_2D_NORM,
            torch.sin(theta) / theta.clamp_min(MIN_2D_NORM),
            torch.zeros_like(theta),
        )
        rays_cam = torch.stack(
            (scale * u, scale * v, torch.cos(theta)),
            dim=-1,
        )

    rotation = viewmat[0, :3, :3]
    translation = viewmat[0, :3, 3]
    rays_d = rays_cam @ rotation  # rotation^T applied to each row vector
    rays_o = (-translation @ rotation).expand(H, W, 3)
    return rays_o.contiguous(), rays_d.contiguous()


def geer_intersections(means, quats, scales, opacities, viewmat, camera_model):
    """Tile association for the eval3d rasterizer, as `rasterization` does it."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.geer._preprocess import _prepare_geer

    tile_width = math.ceil(W / TILE_SIZE)
    tile_height = math.ceil(H / TILE_SIZE)
    _, intersections, _ = _prepare_geer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmats=viewmat,
        Ks=_Ks(),
        width=W,
        height=H,
        tile_size=TILE_SIZE,
        tile_width=tile_width,
        tile_height=tile_height,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        camera_model=camera_model,
        rasterize_mode="classic",
        packed=False,
        radial_coeffs=_radial_coeffs(camera_model),
    )
    _, isect_ids, flatten_ids, pbf_bounds = intersections
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(1, tile_height, tile_width)
    return isect_offsets, flatten_ids, pbf_bounds


def render_kernel(
    scene_tensors, camera_model, rays_o=None, rays_d=None, intersections=None
):
    """Render with the CUDA eval3d path, optionally asking for ray gradients."""
    from gsplat.cuda._wrapper import rasterize_to_pixels_eval3d

    means, quats, scales, opacities, colors = scene_tensors
    viewmat = _viewmat()
    if intersections is None:
        intersections = geer_intersections(
            means, quats, scales, opacities, viewmat, camera_model
        )
    isect_offsets, flatten_ids, pbf_bounds = intersections
    render_colors, render_alphas = rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors[None].contiguous(),
        opacities[None].contiguous(),
        viewmat,
        _Ks(),
        W,
        H,
        TILE_SIZE,
        isect_offsets,
        flatten_ids,
        camera_model=camera_model,
        radial_coeffs=_radial_coeffs(camera_model),
        pbf_bounds=pbf_bounds,
        rays_o=rays_o,
        rays_d=rays_d,
    )
    return render_colors, render_alphas


def render_reference(scene_tensors, rays_o, rays_d, intersections, with_bounds=True):
    """The eval3d compositing in plain PyTorch, kept in step with the kernel.

    Same Gaussian response, same alpha floor, same early out, and by default
    the same per Gaussian pixel bounds.  Those bounds are a hard discontinuity
    that the kernel applies and autograd cannot see, so the reference has to
    apply them too for the comparison to be exact.
    """
    means, quats, scales, opacities, colors = scene_tensors
    _, flatten_ids, pbf_bounds = intersections
    viewmat = _viewmat()

    # The rasterizer sorts by the distance of the camera space mean, so the
    # per tile front to back order is a subsequence of this global order.
    means_cam = means @ viewmat[0, :3, :3].T + viewmat[0, :3, 3]
    order = torch.argsort(means_cam.norm(dim=-1))
    # Gaussians that the preprocessing culled never reach the rasterizer, and
    # their bounds entries are left uninitialised.
    drawn = set(flatten_ids.tolist())

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    num_pixels = rays_o.shape[0]
    rows = torch.arange(H, device=device).repeat_interleave(W)
    cols = torch.arange(W, device=device).repeat(H)
    transmittance = torch.ones(num_pixels, device=device, dtype=torch.float32)
    out = torch.zeros(num_pixels, 3, device=device, dtype=torch.float32)
    finished = torch.zeros(num_pixels, device=device, dtype=torch.bool)

    rotations = _quats_to_rotmats(quats)
    for index in order.tolist():
        if index not in drawn:
            continue
        if with_bounds:
            bounds = pbf_bounds[0, index]
            in_bounds = (
                (cols >= bounds[0])
                & (cols < bounds[1])
                & (rows >= bounds[2])
                & (rows < bounds[3])
            )
        else:
            in_bounds = torch.ones(num_pixels, device=device, dtype=torch.bool)

        # Mt = transpose(R * S) with S the inverse scale matrix.
        Mt = (rotations[index] / scales[index][None, :]).T
        gro = (rays_o - means[index]) @ Mt.T
        grd = rays_d @ Mt.T
        grd_n = grd / grd.norm(dim=-1, keepdim=True)
        gcrod = torch.cross(grd_n, gro, dim=-1)
        alpha = torch.clamp(
            opacities[index] * torch.exp(-0.5 * (gcrod * gcrod).sum(-1)), max=0.99
        )

        contributes = in_bounds & (alpha >= MIN_ALPHA) & (~finished)
        next_transmittance = transmittance * (1.0 - alpha)
        stops = contributes & (next_transmittance <= MIN_T)
        finished = finished | stops
        contributes = contributes & (~stops)

        weight = torch.where(
            contributes, alpha * transmittance, torch.zeros_like(alpha)
        )
        out = out + weight[:, None] * colors[index][None, :]
        transmittance = torch.where(contributes, next_transmittance, transmittance)

    return out.reshape(H, W, 3), (1.0 - transmittance).reshape(H, W, 1)


def _quats_to_rotmats(quats):
    """Same convention as `quat_to_rotmat` in the CUDA code (w, x, y, z)."""
    quats = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = quats.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quats.shape[:-1] + (3, 3))


def cotangent(seed=7):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.rand(H, W, 3, generator=generator, dtype=torch.float32).to(device) - 0.5
    )


@pytest.fixture
def resolution(request, monkeypatch):
    """Override the image size, so ragged tiles get exercised as well.

    A resolution that is not a multiple of the tile size leaves threads outside
    the image.  Their pixel index is clamped to the last pixel, so they must
    never write a ray gradient.
    """
    module = sys.modules[__name__]
    width, height = request.param
    monkeypatch.setattr(module, "W", width)
    monkeypatch.setattr(module, "H", height)
    monkeypatch.setattr(module, "CX", width / 2.0)
    monkeypatch.setattr(module, "CY", height / 2.0)
    return width, height


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_ray_origin_gradient_matches_negated_mean_gradient(camera_model):
    """Shifting every ray origin is the same as shifting every Gaussian back."""
    means, quats, scales, opacities, colors = scene()
    means = means.clone().requires_grad_(True)
    rays_o, rays_d = pixel_rays(camera_model, _viewmat())
    rays_o = rays_o.clone()[None].requires_grad_(True)
    rays_d = rays_d.clone()[None].requires_grad_(True)

    render_colors, _ = render_kernel(
        (means, quats, scales, opacities, colors),
        camera_model,
        rays_o=rays_o,
        rays_d=rays_d,
    )
    (render_colors[0] * cotangent()).sum().backward()

    assert rays_o.grad is not None
    assert rays_d.grad is not None
    summed_rays = rays_o.grad.sum(dim=(0, 1, 2))
    summed_means = means.grad.sum(dim=0)
    print(f"{camera_model}: sum v_rays_o = {summed_rays.tolist()}")
    print(f"{camera_model}: sum v_means  = {summed_means.tolist()}")
    assert summed_rays.abs().max() > 1e-3, "gradient is trivially zero"
    torch.testing.assert_close(summed_rays, -summed_means, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
@pytest.mark.parametrize("resolution", [(64, 48), (61, 45)], indirect=True)
def test_ray_gradients_match_pytorch_reference(camera_model, resolution):
    scene_tensors = scene()
    rays_o, rays_d = pixel_rays(camera_model, _viewmat())
    intersections = geer_intersections(*scene_tensors[:4], _viewmat(), camera_model)

    kernel_rays_o = rays_o.clone()[None].requires_grad_(True)
    kernel_rays_d = rays_d.clone()[None].requires_grad_(True)
    kernel_colors, kernel_alphas = render_kernel(
        scene_tensors,
        camera_model,
        rays_o=kernel_rays_o,
        rays_d=kernel_rays_d,
        intersections=intersections,
    )
    grad_output = cotangent()
    (kernel_colors[0] * grad_output).sum().backward()

    reference_rays_o = rays_o.clone().requires_grad_(True)
    reference_rays_d = rays_d.clone().requires_grad_(True)
    reference_colors, reference_alphas = render_reference(
        scene_tensors, reference_rays_o, reference_rays_d, intersections
    )
    (reference_colors * grad_output).sum().backward()

    color_error = (kernel_colors[0] - reference_colors).abs().max().item()
    alpha_error = (kernel_alphas[0] - reference_alphas).abs().max().item()
    print(f"{camera_model} {W}x{H}: max |colour difference| = {color_error:.3e}")
    print(f"{camera_model} {W}x{H}: max |alpha difference|  = {alpha_error:.3e}")
    assert color_error < 1e-4, "the two renderers do not agree on the image"
    assert alpha_error < 1e-4, "the two renderers do not agree on the alpha"

    # How much the tile bounds actually remove, as a diagnostic.  The bounds
    # are meant to be conservative, so this should stay tiny.
    unbounded, _ = render_reference(
        scene_tensors,
        rays_o.clone(),
        rays_d.clone(),
        intersections,
        with_bounds=False,
    )
    clipped = (unbounded - reference_colors).abs().amax(dim=-1)
    print(
        f"{camera_model} {W}x{H}: tile bounds change {int((clipped > 1e-6).sum())} of "
        f"{H * W} pixels, by at most {clipped.max().item():.3e}"
    )

    for name, kernel_grad, reference_grad in (
        ("v_rays_o", kernel_rays_o.grad[0], reference_rays_o.grad),
        ("v_rays_d", kernel_rays_d.grad[0], reference_rays_d.grad),
    ):
        scale = reference_grad.abs().max().item()
        assert scale > 1e-3, f"{name} is trivially zero"
        error = (kernel_grad - reference_grad).abs().max().item()
        print(
            f"{camera_model} {W}x{H}: max |{name} difference| = {error:.3e} (scale {scale:.3e})"
        )
        # The observed difference is around 1e-5 on a scale of 4, which is
        # float32 rounding plus the fast `__expf` the kernel uses.
        torch.testing.assert_close(
            kernel_grad, reference_grad, rtol=1e-3, atol=1e-4 * scale
        )


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_requesting_ray_gradients_leaves_other_gradients_alone(camera_model):
    def run(with_rays):
        means, quats, scales, opacities, colors = scene()
        tensors = [
            means.clone().requires_grad_(True),
            quats.clone().requires_grad_(True),
            scales.clone().requires_grad_(True),
            opacities.clone().requires_grad_(True),
            colors.clone().requires_grad_(True),
        ]
        rays_o, rays_d = pixel_rays(camera_model, _viewmat())
        rays_o = rays_o.clone()[None].requires_grad_(with_rays)
        rays_d = rays_d.clone()[None].requires_grad_(with_rays)
        render_colors, _ = render_kernel(
            tuple(tensors), camera_model, rays_o=rays_o, rays_d=rays_d
        )
        (render_colors[0] * cotangent()).sum().backward()
        return [tensor.grad for tensor in tensors], rays_o, rays_d

    grads_off, rays_o_off, rays_d_off = run(False)
    grads_off_again, _, _ = run(False)
    grads_on, rays_o_on, rays_d_on = run(True)

    assert rays_o_off.grad is None and rays_d_off.grad is None
    assert rays_o_on.grad is not None and rays_d_on.grad is not None

    # The Gaussian gradients are accumulated with atomics, so two identical
    # calls are not bitwise equal in general.  Asking for ray gradients may
    # therefore not change them by more than the baseline already changes
    # between two runs of its own, give or take float32 rounding.
    names = ["v_means", "v_quats", "v_scales", "v_opacities", "v_colors"]
    for name, first, second, with_rays in zip(
        names, grads_off, grads_off_again, grads_on
    ):
        if torch.equal(first, second) and torch.equal(first, with_rays):
            print(f"{camera_model}: {name} is bitwise identical")
            continue
        noise = (first - second).abs().max().item()
        change = (first - with_rays).abs().max().item()
        magnitude = first.abs().max().item()
        print(
            f"{camera_model}: {name} run to run noise {noise:.3e}, "
            f"change with rays {change:.3e}, magnitude {magnitude:.3e}"
        )
        assert change <= max(noise, 1e-6 * magnitude), f"{name} changed"
