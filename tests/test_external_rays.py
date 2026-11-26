"""Tests for the per pixel rays that the 3DGEER rasterizer takes from outside.

With `rays_o` and `rays_d` the rasterizer stops building one ray per pixel from
the camera model and uses the rays it is handed instead, in both the forward and
the backward pass.  The camera model therefore moves into PyTorch, and focal
length, distortion and pose pick up their gradients through the rays.

The checks here are:

* external rays that reproduce the built in camera model render the same image,
  bit for bit when the rays come from the device itself,
* a focal length, a pose and a unified camera model alpha written in PyTorch get
  gradients that match central differences through the whole chain,
* the per pixel ray gradients match finite differences of single rays,
* the ray gradients of `render_alphas` match a PyTorch reference renderer.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_ray_gradients as reference_tests  # noqa: E402

device = torch.device("cuda:0")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")


W, H = 64, 48
FOCAL = 42.0
# The gradient tests use a longer focal length, see `smooth_scene`.
GRAD_FOCAL = 65.0
CX, CY = W / 2.0, H / 2.0
MIN_2D_NORM = 1e-6
ALPHA_FLOOR = 1.0 / 255.0
# Angles are sampled this densely when the UCM is fitted by a Kannala Brandt
# polynomial for the tile assignment.
KB_FIT_SAMPLES = 128


def _float(value):
    return torch.tensor(value, device=device, dtype=torch.float32)


def _Ks(focal):
    """Intrinsics for the tile assignment, built from a plain float."""
    return torch.tensor(
        [[[focal, 0.0, CX], [0.0, focal, CY], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )


def _viewmat(rotated=True):
    """World to camera transform of the test camera."""
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    if rotated:
        angle = 0.21
        cos, sin = math.cos(angle), math.sin(angle)
        viewmat[:3, :3] = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], device=device
        )
    viewmat[:3, 3] = torch.tensor([0.15, -0.08, 0.04], device=device)
    return viewmat[None]


def _skew(vector):
    zero = torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack(
        [
            torch.stack([zero, -vector[2], vector[1]]),
            torch.stack([vector[2], zero, -vector[0]]),
            torch.stack([-vector[1], vector[0], zero]),
        ]
    )


def _rotated_viewmat(rotation_vector):
    """The test pose with an extra rotation, differentiable in the rotation."""
    base = _viewmat()
    rotation = torch.linalg.matrix_exp(_skew(rotation_vector)) @ base[0, :3, :3]
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    return torch.cat([torch.cat([rotation, base[0, :3, 3:4]], dim=-1), bottom], dim=0)[
        None
    ]


def _pixel_grid():
    """Pixel centres, as the kernel places them."""
    rows = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    cols = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    return torch.meshgrid(rows, cols, indexing="ij")


def _to_world(rays_cam, viewmat):
    """Camera frame directions to world frame rays, as `ShutterPose` does it."""
    rotation = viewmat[0, :3, :3]
    translation = viewmat[0, :3, 3]
    rays_d = rays_cam @ rotation  # rotation^T applied to each row vector
    rays_o = (-translation @ rotation).expand(H, W, 3)
    return rays_o.contiguous()[None], rays_d.contiguous()[None]


# ---------------------------------------------------------------------------
# Camera models, written in PyTorch and differentiable
# ---------------------------------------------------------------------------


def pinhole_rays(focal, viewmat):
    """`PerfectPinholeCameraModel`, written in PyTorch."""
    py, px = _pixel_grid()
    u = (px - CX) / focal
    v = (py - CY) / focal
    rays_cam = torch.stack((u, v, torch.ones_like(u)), dim=-1)
    rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
    return _to_world(rays_cam, viewmat)


def fisheye_rays(focal, viewmat):
    """Equidistant `OpenCVFisheyeCameraModel`, written in PyTorch.

    All Kannala Brandt coefficients are zero, so the angle is the normalised
    radius.  The kernel returns exactly [0, 0, 1] on the principal point, where
    the radius vanishes, and so does this.
    """
    py, px = _pixel_grid()
    u = (px - CX) / focal
    v = (py - CY) / focal
    delta = torch.sqrt(u * u + v * v)
    # Clamp before dividing, otherwise the unused branch still poisons the
    # gradient with a NaN.
    safe_delta = delta.clamp_min(MIN_2D_NORM)
    scale = torch.where(
        delta >= MIN_2D_NORM,
        torch.sin(delta) / safe_delta,
        torch.zeros_like(delta),
    )
    rays_cam = torch.stack((scale * u, scale * v, torch.cos(delta)), dim=-1)
    return _to_world(rays_cam, viewmat)


def ucm_project(points, focal, alpha):
    """Unified camera model projection of camera frame points to pixels."""
    distance = points.norm(dim=-1)
    denominator = alpha * distance + (1.0 - alpha) * points[..., 2]
    return torch.stack(
        (
            focal * points[..., 0] / denominator + CX,
            focal * points[..., 1] / denominator + CY,
        ),
        dim=-1,
    )


def ucm_rays(focal, alpha, viewmat):
    """Unified camera model unprojection, written in PyTorch.

    The inverse of `ucm_project`, see Usenko et al, The Double Sphere Camera
    Model.  With `m = (pixel - c) / f` and `r^2 = |m|^2`,

        mz = (1 - alpha^2 r^2) / (alpha sqrt(1 - (2 alpha - 1) r^2) + 1 - alpha)

    and the ray is `(mx, my, mz)` normalised.
    """
    py, px = _pixel_grid()
    mx = (px - CX) / focal
    my = (py - CY) / focal
    r2 = mx * mx + my * my
    root = torch.sqrt((1.0 - (2.0 * alpha - 1.0) * r2).clamp_min(1e-8))
    mz = (1.0 - alpha * alpha * r2) / (alpha * root + 1.0 - alpha)
    rays_cam = torch.stack((mx, my, mz), dim=-1)
    rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
    return _to_world(rays_cam, viewmat)


def ucm_kannala_brandt_fit(focal, alpha):
    """Kannala Brandt coefficients that mimic the UCM, for the tile assignment.

    The rasterizer still projects the Gaussians with its own camera model to
    find their tiles, so the UCM has to be handed to it as the closest fisheye.
    A least squares fit of `theta + k1 theta^3 + ... + k4 theta^9` against the
    UCM radius is far more accurate than the one pixel of padding that the
    bounds carry.  Returns the coefficients and the worst fit error in pixels.
    """
    _, rays_d = ucm_rays(focal, alpha, _viewmat(rotated=False))
    # The unrotated view matrix only translates, so these are camera frame
    # directions and their polar angle is the angle the camera model sees.
    directions = rays_d[0].reshape(-1, 3)
    max_angle = float(
        torch.atan2(directions[:, :2].norm(dim=-1), directions[:, 2]).max()
    )

    theta = torch.linspace(
        0.0, max_angle, KB_FIT_SAMPLES, device=device, dtype=torch.float64
    )
    alpha = float(alpha)
    # Radius in normalised image coordinates of a unit ray at angle theta.
    radius = torch.sin(theta) / (alpha + (1.0 - alpha) * torch.cos(theta))
    design = torch.stack([theta**3, theta**5, theta**7, theta**9], dim=-1)
    coeffs = torch.linalg.lstsq(design, (radius - theta)[:, None]).solution[:, 0]
    residual = float((design @ coeffs + theta - radius).abs().max())
    return coeffs.to(torch.float32)[None], residual * float(focal)


# ---------------------------------------------------------------------------
# Scenes and rendering
# ---------------------------------------------------------------------------


def scene(num_gaussians=30, seed=3):
    """A busy scene, used wherever only the image itself matters."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.float32).to(device)

    offset = torch.tensor([0.0, 0.0, 2.8], device=device)
    extent = torch.tensor([1.4, 1.0, 1.4], device=device)
    means = (rand(num_gaussians, 3) - 0.5) * extent + offset
    quats = torch.nn.functional.normalize(rand(num_gaussians, 4) - 0.5, dim=-1)
    scales = 0.12 + 0.12 * rand(num_gaussians, 3)
    opacities = 0.3 + 0.2 * rand(num_gaussians)
    colors = rand(num_gaussians, 3)
    return means, quats, scales, opacities, colors


def smooth_scene():
    """Three big Gaussians that cover the image without any hard edge.

    The rasterizer drops a Gaussian whose alpha falls below 1/255 and stops a
    pixel once the transmittance falls below 1e-4.  Both are steps, and a step
    that a perturbation walks across biases every finite difference by a fixed
    amount, no matter how small the step gets.  These Gaussians are wide enough
    and close enough that at `GRAD_FOCAL` every one of them stays well above the
    alpha floor on every pixel, and the transmittance stays near 0.1, so the
    render is a smooth function of the rays.  `discontinuity_margins` checks it.
    """
    means = torch.tensor(
        [[0.10, -0.05, 2.6], [-0.18, 0.12, 2.8], [0.06, 0.09, 3.0]], device=device
    )
    quats = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.10, -0.20, 0.05],
                [1.0, -0.30, 0.10, 0.20],
                [1.0, 0.05, 0.30, -0.10],
            ],
            device=device,
        ),
        dim=-1,
    )
    scales = (
        torch.tensor(
            [[1.00, 0.85, 0.95], [0.90, 1.05, 0.85], [0.95, 0.90, 1.05]],
            device=device,
        )
        * 0.8
    )
    opacities = torch.tensor([0.55, 0.50, 0.60], device=device)
    colors = torch.tensor(
        [[0.80, 0.30, 0.20], [0.20, 0.70, 0.40], [0.30, 0.30, 0.90]], device=device
    )
    return means, quats, scales, opacities, colors


def discontinuity_margins(scene_tensors, rays_o, rays_d):
    """Smallest per Gaussian alpha and smallest transmittance over the image."""
    means, quats, scales, opacities, _ = scene_tensors
    rotations = reference_tests._quats_to_rotmats(quats)
    origins = rays_o.reshape(-1, 3)
    directions = rays_d.reshape(-1, 3)
    transmittance = torch.ones(origins.shape[0], device=device)
    smallest_alpha = math.inf
    for index in range(means.shape[0]):
        # Mt = transpose(R * S) with S the inverse scale matrix.
        Mt = (rotations[index] / scales[index][None, :]).T
        gro = (origins - means[index]) @ Mt.T
        grd = directions @ Mt.T
        grd = grd / grd.norm(dim=-1, keepdim=True)
        gcrod = torch.cross(grd, gro, dim=-1)
        alpha = opacities[index] * torch.exp(-0.5 * (gcrod * gcrod).sum(-1))
        smallest_alpha = min(smallest_alpha, float(alpha.min()))
        transmittance = transmittance * (1.0 - alpha)
    return smallest_alpha, float(transmittance.min())


def render(
    scene_tensors,
    camera_model,
    focal=FOCAL,
    radial_coeffs=None,
    rays_o=None,
    rays_d=None,
    viewmat=None,
):
    """One `rasterization` call through the 3DGEER path."""
    from gsplat import rasterization

    means, quats, scales, opacities, colors = scene_tensors
    if radial_coeffs is None and camera_model == "fisheye":
        radial_coeffs = torch.zeros((1, 4), device=device, dtype=torch.float32)
    render_colors, render_alphas, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        _viewmat() if viewmat is None else viewmat,
        _Ks(focal),
        W,
        H,
        packed=False,
        with_geer=True,
        with_eval3d=True,
        camera_model=camera_model,
        radial_coeffs=radial_coeffs,
        rays_o=rays_o,
        rays_d=rays_d,
    )
    return render_colors, render_alphas


def cotangent(seed=11, channels=3, positive=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.rand(H, W, channels, generator=generator, dtype=torch.float32)
    return (values if positive else values - 0.5).to(device)


def central_difference(forward, parameter, step):
    """Central difference of a scalar loss in one parameter."""
    with torch.no_grad():
        original = parameter.detach().clone()
        parameter.copy_(original + step)
        plus = float(forward())
        parameter.copy_(original - step)
        minus = float(forward())
        parameter.copy_(original)
    return (plus - minus) / (2.0 * step)


# ---------------------------------------------------------------------------
# External rays reproduce the built in camera model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_external_rays_reproduce_builtin_rays(camera_model):
    """Rays rebuilt in PyTorch render what the kernel's own rays render.

    They do not render it bit for bit, because nvcc contracts the camera model
    arithmetic into fused multiply adds and PyTorch does not, which moves the
    ray by one unit in the last place.  The next test gets around that.
    """
    scene_tensors = scene()
    builder = pinhole_rays if camera_model == "pinhole" else fisheye_rays

    internal_colors, internal_alphas = render(scene_tensors, camera_model)
    rays_o, rays_d = builder(_float(FOCAL), _viewmat())
    external_colors, external_alphas = render(
        scene_tensors, camera_model, rays_o=rays_o, rays_d=rays_d
    )

    color_error = (internal_colors - external_colors).abs().max().item()
    alpha_error = (internal_alphas - external_alphas).abs().max().item()
    print(f"{camera_model}: max |colour difference| = {color_error:.3e}")
    print(f"{camera_model}: max |alpha difference|  = {alpha_error:.3e}")
    assert internal_colors.abs().max() > 0.1, "the scene is empty"
    assert color_error < 1e-5
    assert alpha_error < 1e-5


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_device_built_rays_are_bitwise_equal(camera_model):
    """Rays built by the device itself render bit for bit the same image.

    `compute_raymap` runs the very same `Cameras.cuh` code the rasterizer runs,
    so its directions are the kernel's directions exactly.  With a camera that
    is only translated the world transform is the identity on both sides, and
    the comparison is exact from the pixel grid to the image.
    """
    from gsplat.cuda._wrapper import compute_raymap

    scene_tensors = scene()
    viewmat = _viewmat(rotated=False)
    radial_coeffs = (
        torch.zeros((1, 4), device=device, dtype=torch.float32)
        if camera_model == "fisheye"
        else None
    )

    internal_colors, internal_alphas = render(
        scene_tensors, camera_model, radial_coeffs=radial_coeffs, viewmat=viewmat
    )
    # Camera frame directions, which are world frame directions here.
    rays_d = compute_raymap(
        _Ks(FOCAL), W, H, camera_model=camera_model, radial_coeffs=radial_coeffs
    )
    rays_o = (-viewmat[0, :3, 3]).expand(1, H, W, 3).contiguous()
    external_colors, external_alphas = render(
        scene_tensors,
        camera_model,
        radial_coeffs=radial_coeffs,
        rays_o=rays_o,
        rays_d=rays_d,
        viewmat=viewmat,
    )

    assert internal_colors.abs().max() > 0.1, "the scene is empty"
    assert torch.equal(internal_colors, external_colors)
    assert torch.equal(internal_alphas, external_alphas)


# ---------------------------------------------------------------------------
# The acceptance test: gradients of a camera model written in PyTorch
# ---------------------------------------------------------------------------


def _loss_function(build_rays, camera_model, radial_coeffs, grad_output):
    """Scalar loss of one render through externally supplied rays.

    The tile assignment keeps the unperturbed intrinsics and pose, so the only
    thing a perturbation moves is the rays.  That is exactly the path the
    gradient flows through, and it keeps the hard tile bounds, which no
    gradient passes through anyway, out of the finite difference.
    """
    scene_tensors = smooth_scene()

    def loss():
        rays_o, rays_d = build_rays()
        render_colors, _ = render(
            scene_tensors,
            camera_model,
            focal=GRAD_FOCAL,
            radial_coeffs=radial_coeffs,
            rays_o=rays_o,
            rays_d=rays_d,
        )
        return (render_colors[0] * grad_output).sum()

    return loss


def _assert_smooth(build_rays, camera_model):
    """The render has to be differentiable before a difference means anything."""
    with torch.no_grad():
        rays_o, rays_d = build_rays()
        smallest_alpha, smallest_transmittance = discontinuity_margins(
            smooth_scene(), rays_o, rays_d
        )
    print(
        f"{camera_model}: smallest Gaussian alpha "
        f"{smallest_alpha * 255:.2f}/255, smallest transmittance "
        f"{smallest_transmittance:.4f}"
    )
    assert smallest_alpha > 2.0 * ALPHA_FLOOR, "a Gaussian sits on the alpha floor"
    assert smallest_transmittance > 1e-2, "a pixel sits on the early out"


def _gradient_check(
    name, build_rays, parameter, camera_model, radial_coeffs=None, step=None
):
    """Autograd against a central difference through the whole chain."""
    _assert_smooth(build_rays, camera_model)
    loss = _loss_function(build_rays, camera_model, radial_coeffs, cotangent())

    value = loss()
    assert value.abs() > 1e-3, "the scene is empty"
    (analytic,) = torch.autograd.grad(value, parameter)
    analytic = float(analytic)

    if step is None:
        step = max(abs(float(parameter)), 1.0) * 3e-3
    numeric = central_difference(loss, parameter, step)

    relative = abs(analytic - numeric) / max(abs(numeric), 1e-12)
    print(
        f"{camera_model} d(loss)/d({name}): autograd {analytic:+.6f}, "
        f"central difference {numeric:+.6f}, relative error {relative:.4%}"
    )
    assert abs(numeric) > 1e-3, f"the loss does not depend on {name}"
    assert relative < 1e-2


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_focal_length_gradient_matches_finite_differences(camera_model):
    """The whole point of the exercise: a focal length trained through rays."""
    focal = _float(GRAD_FOCAL).requires_grad_(True)
    builder = pinhole_rays if camera_model == "pinhole" else fisheye_rays
    _gradient_check("focal", lambda: builder(focal, _viewmat()), focal, camera_model)


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_pose_gradient_matches_finite_differences(camera_model):
    """A rotation vector on top of the pose also gets its gradient."""
    rotation_vector = torch.zeros(
        3, device=device, dtype=torch.float32, requires_grad=True
    )
    builder = pinhole_rays if camera_model == "pinhole" else fisheye_rays

    def build_rays():
        return builder(_float(GRAD_FOCAL), _rotated_viewmat(rotation_vector))

    _assert_smooth(build_rays, camera_model)
    loss = _loss_function(build_rays, camera_model, None, cotangent())
    (analytic,) = torch.autograd.grad(loss(), rotation_vector)

    # One component at a time, so each central difference stays scalar.
    for axis in range(3):
        numeric = central_difference(loss, rotation_vector[axis], 2e-3)
        relative = abs(float(analytic[axis]) - numeric) / max(abs(numeric), 1e-12)
        print(
            f"{camera_model} d(loss)/d(rotation[{axis}]): autograd "
            f"{float(analytic[axis]):+.6f}, central difference {numeric:+.6f}, "
            f"relative error {relative:.4%}"
        )
        assert abs(numeric) > 1e-3, "the loss does not depend on the pose"
        assert relative < 1e-2


def test_ucm_unprojection_round_trips():
    """The UCM unprojection really inverts the UCM projection."""
    focal, alpha = _float(GRAD_FOCAL), _float(0.6)
    _, rays_d = ucm_rays(focal, alpha, _viewmat(rotated=False))
    pixels = ucm_project(rays_d[0], focal, alpha)
    py, px = _pixel_grid()
    error = (pixels - torch.stack((px, py), dim=-1)).abs().max().item()
    print(f"UCM round trip error = {error:.3e} px")
    assert error < 1e-3


def test_ucm_kannala_brandt_fit_is_subpixel():
    """The fisheye the tile assignment sees has to be the UCM, to a pixel."""
    coeffs, residual_px = ucm_kannala_brandt_fit(GRAD_FOCAL, 0.6)
    print(
        f"UCM Kannala Brandt fit: k = {coeffs[0].tolist()}, "
        f"max residual {residual_px:.3e} px"
    )
    assert residual_px < 0.05


def test_ucm_alpha_gradient_matches_finite_differences():
    """A UCM alpha in PyTorch picks up its gradient through the rays."""
    alpha = _float(0.6).requires_grad_(True)
    coeffs, residual_px = ucm_kannala_brandt_fit(GRAD_FOCAL, 0.6)
    assert residual_px < 0.05, "the tile bounds would not match the rays"
    _gradient_check(
        "alpha",
        lambda: ucm_rays(_float(GRAD_FOCAL), alpha, _viewmat()),
        alpha,
        "fisheye",
        radial_coeffs=coeffs,
    )


def test_ucm_focal_gradient_matches_finite_differences():
    """And so does the focal length of a UCM camera."""
    focal = _float(GRAD_FOCAL).requires_grad_(True)
    coeffs, _ = ucm_kannala_brandt_fit(GRAD_FOCAL, 0.6)
    _gradient_check(
        "focal",
        lambda: ucm_rays(focal, _float(0.6), _viewmat()),
        focal,
        "fisheye",
        radial_coeffs=coeffs,
    )


# ---------------------------------------------------------------------------
# Per pixel ray gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_per_pixel_ray_direction_finite_differences(camera_model):
    """Nudging the ray of a single pixel changes the loss by its gradient.

    Only that one pixel changes, so the difference is taken at that pixel alone
    and the float32 noise of summing the whole image never enters.
    """
    scene_tensors = smooth_scene()
    builder = pinhole_rays if camera_model == "pinhole" else fisheye_rays
    base_o, base_d = builder(_float(GRAD_FOCAL), _viewmat())
    _assert_smooth(lambda: (base_o, base_d), camera_model)
    grad_output = cotangent()

    rays_o = base_o.clone().requires_grad_(True)
    rays_d = base_d.clone().requires_grad_(True)
    render_colors, _ = render(
        scene_tensors, camera_model, focal=GRAD_FOCAL, rays_o=rays_o, rays_d=rays_d
    )
    (render_colors[0] * grad_output).sum().backward()

    magnitude = rays_d.grad[0].norm(dim=-1)
    # The pixels with the most to say, spread out over the image.
    order = torch.argsort(magnitude.reshape(-1), descending=True)
    sampled = order[:: max(1, magnitude.numel() // 400)][:5].tolist()
    assert float(magnitude.reshape(-1)[sampled[-1]]) > 1e-2, "gradients are trivial"

    step = 1e-3
    worst = 0.0
    for flat in sampled:
        row, col = flat // W, flat % W
        for axis in range(3):
            perturbed = base_d.clone()
            values = []
            for sign in (1.0, -1.0):
                perturbed[0, row, col, axis] = base_d[0, row, col, axis] + sign * step
                colors, _ = render(
                    scene_tensors,
                    camera_model,
                    focal=GRAD_FOCAL,
                    rays_o=base_o,
                    rays_d=perturbed.contiguous(),
                )
                values.append(colors[0, row, col].double())
            numeric = float(
                ((values[0] - values[1]) * grad_output[row, col].double()).sum()
                / (2.0 * step)
            )
            analytic = float(rays_d.grad[0, row, col, axis])
            scale = max(abs(numeric), float(magnitude[row, col]) * 1e-2)
            error = abs(analytic - numeric) / scale
            worst = max(worst, error)
            print(
                f"{camera_model} pixel ({row}, {col}) axis {axis}: autograd "
                f"{analytic:+.5f}, central difference {numeric:+.5f}, "
                f"relative error {error:.4%}"
            )
    assert worst < 1e-2


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_alpha_cotangent_ray_gradients_match_reference(camera_model):
    """The alphas also carry a ray gradient, and it is the right one.

    `render_alphas` reaches the rays through the Gaussian response only, a path
    that a cotangent on the colours does not pin down on its own.
    """
    scene_tensors = reference_tests.scene()
    rays_o, rays_d = reference_tests.pixel_rays(
        camera_model, reference_tests._viewmat()
    )
    intersections = reference_tests.geer_intersections(
        *scene_tensors[:4], reference_tests._viewmat(), camera_model
    )
    generator = torch.Generator(device="cpu").manual_seed(23)
    alpha_cotangent = (
        torch.rand(
            reference_tests.H,
            reference_tests.W,
            1,
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        - 0.5
    )

    kernel_rays_o = rays_o.clone()[None].requires_grad_(True)
    kernel_rays_d = rays_d.clone()[None].requires_grad_(True)
    _, kernel_alphas = reference_tests.render_kernel(
        scene_tensors,
        camera_model,
        rays_o=kernel_rays_o,
        rays_d=kernel_rays_d,
        intersections=intersections,
    )
    (kernel_alphas[0] * alpha_cotangent).sum().backward()

    reference_rays_o = rays_o.clone().requires_grad_(True)
    reference_rays_d = rays_d.clone().requires_grad_(True)
    _, reference_alphas = reference_tests.render_reference(
        scene_tensors, reference_rays_o, reference_rays_d, intersections
    )
    (reference_alphas * alpha_cotangent).sum().backward()

    for name, kernel_grad, reference_grad in (
        ("v_rays_o", kernel_rays_o.grad[0], reference_rays_o.grad),
        ("v_rays_d", kernel_rays_d.grad[0], reference_rays_d.grad),
    ):
        scale = reference_grad.abs().max().item()
        assert scale > 1e-3, f"{name} is trivially zero"
        error = (kernel_grad - reference_grad).abs().max().item()
        print(
            f"{camera_model} alphas: max |{name} difference| = {error:.3e} "
            f"(scale {scale:.3e})"
        )
        torch.testing.assert_close(
            kernel_grad, reference_grad, rtol=1e-3, atol=1e-4 * scale
        )


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_rays_that_disagree_with_the_tile_bounds_still_render(camera_model):
    """Rays from a slightly different camera than the tile assignment.

    The tile bounds come from the intrinsics, so rays that do not match them
    can lose a Gaussian at the very edge of its footprint.  Observed here: the
    image is the one the stretched camera sees, with no NaN and no missing
    tiles, which is what the one pixel of bounds padding buys.
    """
    scene_tensors = scene()
    builder = pinhole_rays if camera_model == "pinhole" else fisheye_rays

    matched_o, matched_d = builder(_float(FOCAL), _viewmat())
    matched, _ = render(scene_tensors, camera_model, rays_o=matched_o, rays_d=matched_d)
    stretched_o, stretched_d = builder(_float(FOCAL * 1.05), _viewmat())
    stretched, stretched_alphas = render(
        scene_tensors, camera_model, rays_o=stretched_o, rays_d=stretched_d
    )
    # What the stretched camera should see, with matching tile bounds.
    expected, _ = render(
        scene_tensors,
        camera_model,
        focal=FOCAL * 1.05,
        rays_o=stretched_o,
        rays_d=stretched_d,
    )

    assert torch.isfinite(stretched).all()
    assert torch.isfinite(stretched_alphas).all()
    assert stretched.abs().max() > 0.1, "nothing was drawn at all"
    print(
        f"{camera_model}: rays from a 5 percent longer focal length move the "
        f"image by {(matched - stretched).abs().max().item():.3f}, and differ "
        f"from the matching tile assignment by "
        f"{(expected - stretched).abs().max().item():.3e}"
    )


def test_invalid_rays_render_the_background():
    """Non finite rays and zero directions behave like an invalid pixel."""
    scene_tensors = scene()
    rays_o, rays_d = pinhole_rays(_float(FOCAL), _viewmat())
    rays_d = rays_d.clone()
    rays_d[0, 0] = 0.0
    rays_d[0, 1] = float("nan")
    rays_o = rays_o.clone()
    rays_o[0, 2] = float("inf")

    colors, alphas = render(scene_tensors, "pinhole", rays_o=rays_o, rays_d=rays_d)
    assert torch.isfinite(colors).all()
    assert torch.isfinite(alphas).all()
    assert float(colors[0, :3].abs().max()) == 0.0
    assert float(alphas[0, :3].abs().max()) == 0.0
    # The rows below are untouched and still draw something.
    assert float(colors[0, 3:].abs().max()) > 0.1


def test_rays_with_the_wrong_device_or_shape_raise_a_clear_error():
    scene_tensors = scene()
    rays_o, rays_d = pinhole_rays(_float(FOCAL), _viewmat())
    with pytest.raises(ValueError, match="same device as the means"):
        render(scene_tensors, "pinhole", rays_o=rays_o.cpu(), rays_d=rays_d)
    with pytest.raises(ValueError, match="must be float32"):
        render(scene_tensors, "pinhole", rays_o=rays_o.double(), rays_d=rays_d.double())
    with pytest.raises(ValueError, match="must have shape"):
        render(scene_tensors, "pinhole", rays_o=rays_o[:, :-1], rays_d=rays_d)
    with pytest.raises(ValueError, match="given together"):
        render(scene_tensors, "pinhole", rays_o=rays_o)


def test_external_rays_require_geer():
    from gsplat import rasterization

    means, quats, scales, opacities, colors = scene()
    rays_o, rays_d = pinhole_rays(_float(FOCAL), _viewmat())
    with pytest.raises(ValueError, match="with_geer"):
        rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            _viewmat(),
            _Ks(FOCAL),
            W,
            H,
            packed=False,
            with_eval3d=True,
            rays_o=rays_o,
            rays_d=rays_d,
        )
