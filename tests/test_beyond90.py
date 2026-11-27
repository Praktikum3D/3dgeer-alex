"""Tests for rendering past 90 degrees off axis with 3DGEER.

Two changes are under test here.

The rasterizer used to score a Gaussian by its distance to the *infinite* line
through a pixel ray, so a Gaussian sitting behind the ray origin could still be
drawn at full opacity.  It now scores the half line ``t >= 0``: when the closest
point of the line lies behind the origin, the Gaussian is evaluated at the
origin instead.  Both expressions and their first derivatives agree where the
two branches meet, so the clamp introduces no seam.

The tile assignment used to build a perspective bounding frustum in tangent
space, which needs the Gaussian centre in front of the camera plane.  With
``geer_tile_mode="projection"`` the centres are projected with the camera model
itself, where ``atan2(rho, z)`` keeps working past 90 degrees, their pixel
footprint is bounded from the 3 sigma ball, and the stock tile assignment takes
over.  That is what lets a fisheye render its full image circle.
"""

import math
import time

import pytest
import torch

device = torch.device("cuda:0")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")


# A fisheye wide enough that 170 degrees still lands inside the image.
WIDE_W = WIDE_H = 256
WIDE_FOCAL = 42.0
# A smaller camera for the gradient checks, where every render is repeated.
W, H = 64, 48
FOCAL = 42.0
GRAD_FOCAL = 65.0
MIN_2D_NORM = 1e-6
ALPHA_FLOOR = 1.0 / 255.0


def _Ks(focal, width, height):
    return torch.tensor(
        [[[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )


def _identity_viewmat():
    return torch.eye(4, device=device, dtype=torch.float32)[None]


def _zero_radial():
    """All Kannala Brandt coefficients zero, the equidistant fisheye."""
    return torch.zeros((1, 4), device=device, dtype=torch.float32)


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
    ).reshape(-1, 3, 3)


def equidistant_rays(focal, width, height, viewmat=None):
    """Equidistant fisheye rays, written in PyTorch and differentiable."""
    rows = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    cols = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    py, px = torch.meshgrid(rows, cols, indexing="ij")
    u = (px - 0.5 * width) / focal
    v = (py - 0.5 * height) / focal
    delta = torch.sqrt(u * u + v * v)
    safe_delta = delta.clamp_min(MIN_2D_NORM)
    scale = torch.where(
        delta >= MIN_2D_NORM,
        torch.sin(delta) / safe_delta,
        torch.zeros_like(delta),
    )
    rays_cam = torch.stack((scale * u, scale * v, torch.cos(delta)), dim=-1)
    if viewmat is None:
        origin = torch.zeros(3, device=device, dtype=torch.float32)
        rays_d = rays_cam
        rays_o = origin.expand(height, width, 3)
    else:
        rotation = viewmat[0, :3, :3]
        translation = viewmat[0, :3, 3]
        rays_d = rays_cam @ rotation
        rays_o = (-translation @ rotation).expand(height, width, 3)
    return rays_o.contiguous()[None], rays_d.contiguous()[None]


def render(
    scene_tensors,
    geer_tile_mode,
    focal=WIDE_FOCAL,
    width=WIDE_W,
    height=WIDE_H,
    camera_model="fisheye",
    viewmat=None,
    rays_o=None,
    rays_d=None,
    render_mode="RGB",
):
    """One `rasterization` call through the 3DGEER path."""
    from gsplat import rasterization

    means, quats, scales, opacities, colors = scene_tensors
    return rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        _identity_viewmat() if viewmat is None else viewmat,
        _Ks(focal, width, height),
        width,
        height,
        packed=False,
        with_geer=True,
        with_eval3d=True,
        camera_model=camera_model,
        radial_coeffs=_zero_radial() if camera_model == "fisheye" else None,
        geer_tile_mode=geer_tile_mode,
        rays_o=rays_o,
        rays_d=rays_d,
        render_mode=render_mode,
    )


def one_gaussian(theta_deg, distance=3.0, scale=0.05, opacity=1.0, azimuth_deg=0.0):
    """A small isotropic Gaussian `theta_deg` off the optical axis."""
    theta = math.radians(theta_deg)
    phi = math.radians(azimuth_deg)
    means = torch.tensor(
        [
            [
                distance * math.sin(theta) * math.cos(phi),
                distance * math.sin(theta) * math.sin(phi),
                distance * math.cos(theta),
            ]
        ],
        device=device,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.full((1, 3), scale, device=device)
    opacities = torch.full((1,), opacity, device=device)
    colors = torch.ones((1, 3), device=device)
    return means, quats, scales, opacities, colors


def equidistant_pixel(theta_deg, azimuth_deg=0.0, focal=WIDE_FOCAL, width=WIDE_W,
                      height=WIDE_H):
    """Where an equidistant fisheye puts a direction, analytically."""
    radius = focal * math.radians(theta_deg)
    return (
        0.5 * width + radius * math.cos(math.radians(azimuth_deg)),
        0.5 * height + radius * math.sin(math.radians(azimuth_deg)),
    )


def clamped_alpha(scene_tensors, rays_o, rays_d):
    """The half line response of every Gaussian, in PyTorch.

    This is the reference for what the kernel now computes, and the branch mask
    it returns says which pixels take the clamped branch.
    """
    means, quats, scales, opacities, _ = scene_tensors
    rotations = _quats_to_rotmats(quats)
    origins = rays_o.reshape(-1, 3)
    directions = rays_d.reshape(-1, 3)
    alphas, clamped = [], []
    for index in range(means.shape[0]):
        # Mt = transpose(R * S) with S the inverse scale matrix.
        Mt = (rotations[index] / scales[index][None, :]).T
        gro = (origins - means[index]) @ Mt.T
        grd = directions @ Mt.T
        grd_n = grd / grd.norm(dim=-1, keepdim=True)
        tstar = -(gro * grd_n).sum(-1)
        gcrod = torch.cross(grd_n, gro, dim=-1)
        line = (gcrod * gcrod).sum(-1)
        half_line = (gro * gro).sum(-1)
        is_clamped = tstar <= 0.0
        power = -0.5 * torch.where(is_clamped, half_line, line)
        alphas.append(opacities[index] * torch.exp(power))
        clamped.append(is_clamped)
    return torch.stack(alphas), torch.stack(clamped)


def central_difference(forward, parameter, step):
    with torch.no_grad():
        original = parameter.detach().clone()
        parameter.copy_(original + step)
        plus = float(forward())
        parameter.copy_(original - step)
        minus = float(forward())
        parameter.copy_(original)
    return (plus - minus) / (2.0 * step)


def cotangent(height, width, channels=3, seed=5):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.rand(height, width, channels, generator=generator)
    return (values - 0.5).to(device)


# ---------------------------------------------------------------------------
# 1. A Gaussian past 90 degrees lands where the camera model says it should
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta_deg", [95.0, 120.0, 170.0])
def test_gaussian_beyond_90_degrees_renders_at_the_right_pixel(theta_deg):
    """The projection path draws it; the frustum path drops it entirely.

    The second half is the point of the exercise: with `geer_tile_mode="pbf"`
    the centre sits behind the camera plane, `in_frustum` culls it, and the
    pixel stays background no matter how exact the per ray evaluation is.
    """
    scene_tensors = one_gaussian(theta_deg, azimuth_deg=25.0)
    expected_x, expected_y = equidistant_pixel(theta_deg, azimuth_deg=25.0)

    _, alphas, _ = render(scene_tensors, "projection")
    image = alphas[0, ..., 0]
    flat = int(image.reshape(-1).argmax())
    row, col = flat // WIDE_W, flat % WIDE_W
    error = math.hypot(col + 0.5 - expected_x, row + 0.5 - expected_y)
    print(
        f"theta {theta_deg:5.1f} deg: brightest pixel ({col}, {row}), expected "
        f"({expected_x:.1f}, {expected_y:.1f}), error {error:.2f} px, alpha "
        f"{float(image.max()):.4f}"
    )
    assert float(image.max()) > 0.5, "the Gaussian was not drawn"
    assert error <= 2.0

    _, pbf_alphas, _ = render(scene_tensors, "pbf")
    print(f"theta {theta_deg:5.1f} deg: 'pbf' draws {float(pbf_alphas.max()):.4f}")
    assert float(pbf_alphas.max()) == 0.0, "the old path should see nothing here"


def test_below_90_degrees_lands_at_the_right_pixel_in_both_modes():
    """The control: inside the old field of view both paths agree."""
    for theta_deg in (30.0, 60.0, 85.0):
        scene_tensors = one_gaussian(theta_deg, azimuth_deg=-40.0)
        expected_x, expected_y = equidistant_pixel(theta_deg, azimuth_deg=-40.0)
        for mode in ("pbf", "projection"):
            _, alphas, _ = render(scene_tensors, mode)
            image = alphas[0, ..., 0]
            flat = int(image.reshape(-1).argmax())
            row, col = flat // WIDE_W, flat % WIDE_W
            error = math.hypot(col + 0.5 - expected_x, row + 0.5 - expected_y)
            print(f"theta {theta_deg:5.1f} deg, {mode:10s}: error {error:.2f} px")
            assert float(image.max()) > 0.5
            assert error <= 2.0


# ---------------------------------------------------------------------------
# 2. Nothing behind the ray origin is drawn
# ---------------------------------------------------------------------------


def test_nothing_behind_the_ray_origin_is_drawn():
    """A Gaussian on the backward extension of a ray stays invisible.

    The Gaussian is a thin disk at `(0, 0, -2)`, spread wide in x and y and
    narrow in z, so the forward rays miss it by many sigma while its 3 sigma
    ball still swallows the camera and puts it into every tile.  The old
    infinite line distance scored it at full opacity on the optical axis, which
    the reference value printed below shows; the half line scores it at zero.
    Rays that really do point backwards, past 90 degrees, still see it, and so
    they should.
    """
    means = torch.tensor([[0.0, 0.0, -2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[3.0, 3.0, 0.05]], device=device)
    opacities = torch.ones(1, device=device)
    colors = torch.ones((1, 3), device=device)
    scene_tensors = (means, quats, scales, opacities, colors)

    rays_o, rays_d = equidistant_rays(WIDE_FOCAL, WIDE_W, WIDE_H)
    theta = torch.atan2(rays_d[0, ..., :2].norm(dim=-1), rays_d[0, ..., 2])
    forward = theta < math.radians(80.0)
    assert bool(forward.any())

    # What the old infinite line distance would have drawn on the optical axis.
    line_alpha = float(
        torch.exp(
            -0.5
            * torch.cross(
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
                torch.tensor([[0.0, 0.0, 2.0 / 0.05]], device=device),
                dim=-1,
            )
            .square()
            .sum(-1)
        )
    )
    print(f"the infinite line would have drawn alpha {line_alpha:.4f} on axis")
    assert line_alpha > 0.99

    for mode in ("pbf", "projection"):
        _, alphas, _ = render(scene_tensors, mode)
        image = alphas[0, ..., 0]
        print(
            f"{mode:10s}: max alpha on the forward rays "
            f"{float(image[forward].max()):.3e}"
        )
        assert float(image[forward].max()) < ALPHA_FLOOR

    # The backward rays do pass through it, which is not a bug.
    _, alphas, _ = render(scene_tensors, "projection")
    backward = theta > math.radians(150.0)
    print(f"backward rays see alpha {float(alphas[0, ..., 0][backward].max()):.4f}")
    assert float(alphas[0, ..., 0][backward].max()) > 0.5


# ---------------------------------------------------------------------------
# 3. Gradients of the clamped branch
# ---------------------------------------------------------------------------


def clamped_branch_scene():
    """One Gaussian beside the camera, unsaturated and above the alpha floor.

    Its centre sits behind the camera plane and to the side, so every forward
    ray has its closest approach behind the ray origin and takes the clamped
    branch, while the response at the origin is still a healthy alpha. It is
    rotated and anisotropic, so the quaternion and all three scales matter.
    """
    means = torch.tensor([[0.5, 0.0, -0.3]], device=device)
    quats = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.15, -0.25, 0.1]], device=device), dim=-1
    )
    scales = torch.tensor([[0.50, 0.42, 0.60]], device=device)
    opacities = torch.full((1,), 0.5, device=device)
    colors = torch.tensor([[0.8, 0.4, 0.2]], device=device)
    return means, quats, scales, opacities, colors


def clamped_branch_rays():
    """A narrow forward ray bundle, so every pixel takes the clamped branch.

    The origins are spread over a small patch as well, so the clamped response
    varies from pixel to pixel instead of being one constant.
    """
    rows = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    cols = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    py, px = torch.meshgrid(rows, cols, indexing="ij")
    u = (px - 0.5 * W) / (4.0 * FOCAL)
    v = (py - 0.5 * H) / (4.0 * FOCAL)
    rays_cam = torch.stack((u, v, torch.ones_like(u)), dim=-1)
    rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
    rays_o = torch.stack(
        (
            0.25 * (px - 0.5 * W) / (0.5 * W),
            0.25 * (py - 0.5 * H) / (0.5 * H),
            torch.zeros_like(px),
        ),
        dim=-1,
    )
    return rays_o.contiguous()[None], rays_cam.contiguous()[None]


def test_clamped_branch_is_the_one_under_test():
    """Every pixel of the gradient scene really does take the clamped branch."""
    scene_tensors = clamped_branch_scene()
    rays_o, rays_d = clamped_branch_rays()
    alphas, clamped = clamped_alpha(scene_tensors, rays_o[0], rays_d[0])
    print(
        f"clamped pixels {int(clamped.sum())}/{clamped.numel()}, alpha in "
        f"[{float(alphas.min()):.4f}, {float(alphas.max()):.4f}]"
    )
    assert bool(clamped.all()), "the scene does not exercise the clamped branch"
    assert float(alphas.min()) > 4.0 * ALPHA_FLOOR, "a pixel sits on the alpha floor"
    assert float(alphas.max()) < 0.9, "a pixel saturates the alpha clamp"

    # And the kernel agrees with the PyTorch half line reference.
    _, kernel_alphas, _ = render(
        scene_tensors,
        "projection",
        focal=FOCAL,
        width=W,
        height=H,
        rays_o=rays_o,
        rays_d=rays_d,
    )
    error = (
        (kernel_alphas[0, ..., 0].reshape(-1) - alphas[0]).abs().max().item()
    )
    print(f"kernel against the PyTorch half line reference: {error:.3e}")
    assert error < 1e-5


@pytest.mark.parametrize("name", ["means", "quats", "scales", "opacities"])
def test_clamped_branch_gradients_match_finite_differences(name):
    """Autograd through the clamped branch against central differences."""
    scene_tensors = list(clamped_branch_scene())
    index = {"means": 0, "quats": 1, "scales": 2, "opacities": 3}[name]
    parameter = scene_tensors[index].clone().requires_grad_(True)
    scene_tensors[index] = parameter
    rays_o, rays_d = clamped_branch_rays()
    grad_output = cotangent(H, W)

    def loss():
        colors, _, _ = render(
            tuple(scene_tensors),
            "projection",
            focal=FOCAL,
            width=W,
            height=H,
            rays_o=rays_o,
            rays_d=rays_d,
        )
        return (colors[0] * grad_output).sum()

    value = loss()
    assert value.abs() > 1e-3, "the scene is empty"
    (analytic,) = torch.autograd.grad(value, parameter)

    worst = 0.0
    flat = parameter.reshape(-1)
    for position in range(flat.numel()):
        numeric = central_difference(loss, flat[position], 2e-3)
        expected = float(analytic.reshape(-1)[position])
        error = abs(expected - numeric) / max(abs(numeric), 1e-3)
        worst = max(worst, error)
        print(
            f"{name}[{position}]: autograd {expected:+.6f}, central difference "
            f"{numeric:+.6f}, relative error {error:.4%}"
        )
    assert worst < 2e-2


def test_clamped_branch_returns_no_ray_direction_gradient():
    """In the clamped branch the response does not see the direction at all."""
    scene_tensors = clamped_branch_scene()
    base_o, base_d = clamped_branch_rays()
    rays_o = base_o.clone().requires_grad_(True)
    rays_d = base_d.clone().requires_grad_(True)
    grad_output = cotangent(H, W)

    colors, _, _ = render(
        scene_tensors,
        "projection",
        focal=FOCAL,
        width=W,
        height=H,
        rays_o=rays_o,
        rays_d=rays_d,
    )
    (colors[0] * grad_output).sum().backward()

    print(
        f"max |v_rays_d| = {float(rays_d.grad.abs().max()):.3e}, "
        f"max |v_rays_o| = {float(rays_o.grad.abs().max()):.3e}"
    )
    assert float(rays_d.grad.abs().max()) == 0.0
    assert float(rays_o.grad.abs().max()) > 1e-3, "v_rays_o is trivially zero"

    # And `v_rays_o` is the gradient it should be.
    worst = 0.0
    magnitude = rays_o.grad[0].norm(dim=-1).reshape(-1)
    order = torch.argsort(magnitude, descending=True)
    sampled = order[:: max(1, magnitude.numel() // 200)][:4].tolist()
    step = 1e-3
    for position in sampled:
        row, col = position // W, position % W
        for axis in range(3):
            perturbed = base_o.clone()
            values = []
            for sign in (1.0, -1.0):
                perturbed[0, row, col, axis] = (
                    base_o[0, row, col, axis] + sign * step
                )
                image, _, _ = render(
                    scene_tensors,
                    "projection",
                    focal=FOCAL,
                    width=W,
                    height=H,
                    rays_o=perturbed.contiguous(),
                    rays_d=base_d,
                )
                values.append(image[0, row, col].double())
            numeric = float(
                ((values[0] - values[1]) * grad_output[row, col].double()).sum()
                / (2.0 * step)
            )
            analytic = float(rays_o.grad[0, row, col, axis])
            scale = max(abs(numeric), float(magnitude[position]) * 1e-2)
            error = abs(analytic - numeric) / scale
            worst = max(worst, error)
            print(
                f"v_rays_o ({row}, {col}) axis {axis}: autograd {analytic:+.6f}, "
                f"central difference {numeric:+.6f}, relative error {error:.4%}"
            )
    assert worst < 1e-2


def test_the_clamp_seam_is_smooth():
    """Walking a camera through the seam leaves no step in the image.

    The ray origin slides along z past an off axis Gaussian, so the closest
    point of every ray changes sign in the middle of the sweep.  Both branches
    agree in value and in slope there, so the image has to move smoothly, and
    halving the step has to halve the largest change between two frames.
    """
    means = torch.tensor([[0.45, 0.0, 0.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.full((1, 3), 0.3, device=device)
    opacities = torch.full((1,), 0.8, device=device)
    colors = torch.ones((1, 3), device=device)
    scene_tensors = (means, quats, scales, opacities, colors)

    _, base_d = clamped_branch_rays()

    def sweep(num_steps):
        span = 0.4
        images = []
        for step in range(num_steps + 1):
            z = -0.5 * span + span * step / num_steps
            rays_o = torch.zeros(1, H, W, 3, device=device)
            rays_o[..., 2] = z
            image, _, _ = render(
                scene_tensors,
                "projection",
                focal=FOCAL,
                width=W,
                height=H,
                rays_o=rays_o.contiguous(),
                rays_d=base_d,
            )
            images.append(image[0].clone())
        jumps = [
            float((images[i + 1] - images[i]).abs().max()) for i in range(num_steps)
        ]
        return max(jumps)

    # The seam really is inside the sweep.
    rays_o = torch.zeros(1, H, W, 3, device=device)
    _, clamped_start = clamped_alpha(scene_tensors, rays_o[0] - 0.2, base_d[0])
    _, clamped_end = clamped_alpha(scene_tensors, rays_o[0] + 0.2, base_d[0])
    assert not bool(clamped_start.any()), "the sweep starts on the clamped side"
    assert bool(clamped_end.all()), "the sweep does not reach the clamped side"

    coarse = sweep(20)
    fine = sweep(40)
    print(f"largest frame to frame change: coarse {coarse:.4e}, fine {fine:.4e}")
    assert coarse > 1e-3, "the sweep does not change the image at all"
    assert fine < 0.62 * coarse, "halving the step did not halve the change"


# ---------------------------------------------------------------------------
# 4. The usual gradients still work in projection mode
# ---------------------------------------------------------------------------


def smooth_scene():
    """Three big Gaussians that cover the image without any hard edge."""
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


def _assert_smooth(scene_tensors, rays_o, rays_d):
    alphas, _ = clamped_alpha(scene_tensors, rays_o, rays_d)
    transmittance = torch.prod(1.0 - alphas, dim=0)
    print(
        f"smallest Gaussian alpha {float(alphas.min()) * 255:.2f}/255, smallest "
        f"transmittance {float(transmittance.min()):.4f}"
    )
    assert float(alphas.min()) > 2.0 * ALPHA_FLOOR
    assert float(transmittance.min()) > 1e-2


@pytest.mark.parametrize("name", ["means", "quats", "scales", "opacities"])
def test_gaussian_gradients_in_projection_mode(name):
    """The ordinary Gaussian gradients survive the new tile assignment."""
    scene_tensors = list(smooth_scene())
    index = {"means": 0, "quats": 1, "scales": 2, "opacities": 3}[name]
    parameter = scene_tensors[index].clone().requires_grad_(True)
    scene_tensors[index] = parameter
    rays_o, rays_d = equidistant_rays(GRAD_FOCAL, W, H)
    _assert_smooth(smooth_scene(), rays_o[0], rays_d[0])
    grad_output = cotangent(H, W)

    def loss():
        colors, _, _ = render(
            tuple(scene_tensors),
            "projection",
            focal=GRAD_FOCAL,
            width=W,
            height=H,
            rays_o=rays_o,
            rays_d=rays_d,
        )
        return (colors[0] * grad_output).sum()

    (analytic,) = torch.autograd.grad(loss(), parameter)
    generator = torch.Generator(device="cpu").manual_seed(7)
    flat = parameter.reshape(-1)
    positions = torch.randperm(flat.numel(), generator=generator)[:5].tolist()

    worst = 0.0
    for position in positions:
        numeric = central_difference(loss, flat[position], 2e-3)
        expected = float(analytic.reshape(-1)[position])
        error = abs(expected - numeric) / max(abs(numeric), 1e-3)
        worst = max(worst, error)
        print(
            f"{name}[{position}]: autograd {expected:+.6f}, central difference "
            f"{numeric:+.6f}, relative error {error:.4%}"
        )
    assert worst < 2e-2


def test_focal_length_gradient_in_projection_mode():
    """The whole chain, focal length to rays to image, in projection mode."""
    focal = torch.tensor(GRAD_FOCAL, device=device).requires_grad_(True)
    scene_tensors = smooth_scene()
    grad_output = cotangent(H, W)
    _assert_smooth(scene_tensors, *[t[0] for t in equidistant_rays(focal.detach(), W, H)])

    def loss():
        rays_o, rays_d = equidistant_rays(focal, W, H)
        colors, _, _ = render(
            scene_tensors,
            "projection",
            focal=GRAD_FOCAL,
            width=W,
            height=H,
            rays_o=rays_o,
            rays_d=rays_d,
        )
        return (colors[0] * grad_output).sum()

    value = loss()
    (analytic,) = torch.autograd.grad(value, focal)
    numeric = central_difference(loss, focal, GRAD_FOCAL * 3e-3)
    relative = abs(float(analytic) - numeric) / max(abs(numeric), 1e-12)
    print(
        f"d(loss)/d(focal): autograd {float(analytic):+.6f}, central difference "
        f"{numeric:+.6f}, relative error {relative:.4%}"
    )
    assert abs(numeric) > 1e-3
    assert relative < 1e-2


# ---------------------------------------------------------------------------
# 5. Below 90 degrees both tile assignments render the same image
# ---------------------------------------------------------------------------


def narrow_scene(num_gaussians=150, seed=4):
    """A busy scene that stays well inside 80 degrees of the optical axis."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.float32).to(device)

    offset = torch.tensor([0.0, 0.0, 2.8], device=device)
    extent = torch.tensor([1.6, 1.2, 1.6], device=device)
    means = (rand(num_gaussians, 3) - 0.5) * extent + offset
    quats = torch.nn.functional.normalize(rand(num_gaussians, 4) - 0.5, dim=-1)
    scales = 0.10 + 0.12 * rand(num_gaussians, 3)
    opacities = 0.25 + 0.25 * rand(num_gaussians)
    colors = rand(num_gaussians, 3)
    return means, quats, scales, opacities, colors


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
def test_the_two_tile_modes_agree_below_90_degrees(camera_model):
    """Same image from both paths, but not bit for bit, and here is why.

    The two paths sort the Gaussians by different keys, camera frame z against
    euclidean distance, so two Gaussians that overlap in a pixel can swap
    places in the blend.  The alpha compositing of two overlapping Gaussians is
    not commutative, which leaves a difference of the order of the product of
    their alphas.  On top of that the frustum path carries per Gaussian pixel
    bounds, which cut off the far tail of a footprint that the projection path
    still evaluates.  Both effects are small, neither is zero.
    """
    scene_tensors = narrow_scene()
    means = scene_tensors[0]
    theta = torch.atan2(means[:, :2].norm(dim=-1), means[:, 2]).max()
    assert float(theta) < math.radians(80.0)

    reference, reference_alphas, _ = render(
        scene_tensors, "pbf", focal=FOCAL, width=W, height=H,
        camera_model=camera_model,
    )
    new, new_alphas, _ = render(
        scene_tensors, "projection", focal=FOCAL, width=W, height=H,
        camera_model=camera_model,
    )

    assert float(reference.abs().max()) > 0.1, "the scene is empty"
    mean_error = float((reference - new).abs().mean())
    max_error = float((reference - new).abs().max())
    alpha_error = float((reference_alphas - new_alphas).abs().max())
    print(
        f"{camera_model}: mean |difference| {mean_error:.3e}, max "
        f"{max_error:.3e}, max |alpha difference| {alpha_error:.3e}"
    )
    assert mean_error < 1e-4
    assert max_error < 1e-2
    assert alpha_error < 1e-2


def test_projection_mode_reports_euclidean_depth():
    """`meta["depths"]` and the "D" render mode carry distances, not z."""
    scene_tensors = one_gaussian(40.0, distance=3.0, scale=0.2)
    _, _, meta = render(scene_tensors, "projection")
    distance = float(scene_tensors[0].norm())
    print(f"depth {float(meta['depths'][0, 0]):.4f}, distance {distance:.4f}")
    assert abs(float(meta["depths"][0, 0]) - distance) < 1e-4
    assert meta["pbf_bounds"] is None

    _, _, pbf_meta = render(scene_tensors, "pbf")
    camera_z = float(scene_tensors[0][0, 2])
    assert abs(float(pbf_meta["depths"][0, 0]) - camera_z) < 1e-4


def test_projection_mode_renders_one_image_per_call():
    """Several images at once are refused, and here is why.

    The maths of the new path takes any number of cameras, but the tile
    assignment it shares with the rest of gsplat does not. `isect_tiles` fills
    a tile range table sized for a single image while `identify_tile_ranges`
    indexes it with the image id folded into the key, so two or more images
    write past the end of that table and corrupt whatever tensor the caching
    allocator put next to it. The table is dead output that nothing reads, but
    the fix belongs to `IntersectTile.cu` and not to this branch, so the path
    refuses the case instead of silently scribbling over memory.
    """
    from gsplat import rasterization

    scene_tensors = narrow_scene()
    viewmats = torch.eye(4, device=device)[None].repeat(3, 1, 1)
    Ks = _Ks(FOCAL, W, H).repeat(3, 1, 1)
    radial = _zero_radial().repeat(3, 1)

    with pytest.raises(ValueError, match="one image at a time"):
        rasterization(
            *scene_tensors,
            viewmats,
            Ks,
            W,
            H,
            packed=False,
            with_geer=True,
            with_eval3d=True,
            camera_model="fisheye",
            radial_coeffs=radial,
            geer_tile_mode="projection",
        )


def test_projection_mode_rejects_what_it_cannot_model():
    from gsplat import rasterization

    scene_tensors = one_gaussian(30.0)
    with pytest.raises(ValueError, match="camera models only"):
        render(scene_tensors, "projection", camera_model="ortho")
    with pytest.raises(ValueError, match="needs `with_geer=True`"):
        rasterization(
            *scene_tensors,
            _identity_viewmat(),
            _Ks(WIDE_FOCAL, WIDE_W, WIDE_H),
            WIDE_W,
            WIDE_H,
            packed=False,
            with_eval3d=True,
            geer_tile_mode="projection",
        )


# ---------------------------------------------------------------------------
# 6. External rays past 90 degrees, with a focal length gradient
# ---------------------------------------------------------------------------


def test_external_rays_beyond_90_degrees_carry_a_focal_gradient():
    """The purpose of the whole branch, in one test.

    Rays for the full image circle of a 195 degree fisheye are built in
    PyTorch, a Gaussian sits 120 degrees off axis, and the focal length that
    made the rays picks up a gradient from the pixels that see it.
    """
    focal = torch.tensor(WIDE_FOCAL, device=device).requires_grad_(True)
    scene_tensors = one_gaussian(120.0, distance=2.5, scale=0.25, opacity=0.8)
    expected_x, expected_y = equidistant_pixel(120.0)
    grad_output = cotangent(WIDE_H, WIDE_W, seed=9)

    def loss():
        rays_o, rays_d = equidistant_rays(focal, WIDE_W, WIDE_H)
        colors, alphas, _ = render(
            scene_tensors, "projection", rays_o=rays_o, rays_d=rays_d
        )
        return (colors[0] * grad_output).sum(), alphas

    value, alphas = loss()
    image = alphas[0, ..., 0]
    flat = int(image.reshape(-1).argmax())
    row, col = flat // WIDE_W, flat % WIDE_W
    error = math.hypot(col + 0.5 - expected_x, row + 0.5 - expected_y)
    print(
        f"external rays, 120 deg: brightest pixel ({col}, {row}), expected "
        f"({expected_x:.1f}, {expected_y:.1f}), error {error:.2f} px"
    )
    assert float(image.max()) > 0.5
    assert error <= 2.0

    (analytic,) = torch.autograd.grad(value, focal)
    numeric = central_difference(lambda: loss()[0], focal, WIDE_FOCAL * 2e-3)
    relative = abs(float(analytic) - numeric) / max(abs(numeric), 1e-12)
    print(
        f"d(loss)/d(focal) past 90 degrees: autograd {float(analytic):+.6f}, "
        f"central difference {numeric:+.6f}, relative error {relative:.4%}"
    )
    assert abs(numeric) > 1e-3, "the loss does not depend on the focal length"
    assert relative < 5e-2


# ---------------------------------------------------------------------------
# 7. Timing
# ---------------------------------------------------------------------------


def test_report_time_per_view():
    """No assertion, just the numbers the report asks for."""
    generator = torch.Generator(device="cpu").manual_seed(1)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.float32).to(device)

    num_gaussians = 6000
    means = (rand(num_gaussians, 3) - 0.5) * torch.tensor(
        [2.0, 2.0, 2.0], device=device
    ) + torch.tensor([0.0, 0.0, 3.0], device=device)
    quats = torch.nn.functional.normalize(rand(num_gaussians, 4) - 0.5, dim=-1)
    scales = 0.02 + 0.03 * rand(num_gaussians, 3)
    opacities = 0.2 + 0.3 * rand(num_gaussians)
    colors = rand(num_gaussians, 3)
    scene_tensors = (means, quats, scales, opacities, colors)

    for camera_model in ("pinhole", "fisheye"):
        for mode in ("pbf", "projection"):
            for _ in range(5):
                render(
                    scene_tensors, mode, focal=160.0, width=256, height=256,
                    camera_model=camera_model,
                )
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(30):
                render(
                    scene_tensors, mode, focal=160.0, width=256, height=256,
                    camera_model=camera_model,
                )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 30.0
            print(f"{camera_model:8s} {mode:10s}: {elapsed * 1e3:.2f} ms per view")
