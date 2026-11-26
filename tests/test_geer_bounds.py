"""Tests for the 3DGEER pixel bounds used for tile association.

The fisheye bounds helper used to floor both edges of the projected footprint
while every consumer treats the bounds as half open.  That dropped the last
column and row of every Gaussian footprint, made image column ``W - 1`` and row
``H - 1`` unreachable, and deleted Gaussians whose projection fell inside a
single pixel.  These tests pin the expected behaviour down.
"""

import math

import pytest
import torch

device = torch.device("cuda:0")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")


W, H = 64, 48
FX = FY = 40.0
CX, CY = W / 2.0, H / 2.0
DEPTH = 2.0


def _Ks():
    return torch.tensor(
        [[[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )


def _viewmat():
    return torch.eye(4, device=device, dtype=torch.float32)[None]


def ray_direction(camera_model, px, py):
    """Direction of the ray through the centre of pixel (px, py), camera frame."""
    u = (px + 0.5 - CX) / FX
    v = (py + 0.5 - CY) / FY
    if camera_model == "pinhole":
        return torch.tensor([u, v, 1.0], device=device, dtype=torch.float32)
    # Equidistant fisheye: all Kannala Brandt coefficients are zero.
    theta = math.hypot(u, v)
    if theta == 0.0:
        return torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    return torch.tensor(
        [math.sin(theta) * u / theta, math.sin(theta) * v / theta, math.cos(theta)],
        device=device,
        dtype=torch.float32,
    )


def render(means, quats, scales, opacities, colors, camera_model):
    from gsplat import rasterization

    radial_coeffs = None
    if camera_model == "fisheye":
        # All zeros is the equidistant Kannala Brandt model.
        radial_coeffs = torch.zeros((1, 4), device=device, dtype=torch.float32)
    rendered, alphas, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        _viewmat(),
        _Ks(),
        W,
        H,
        packed=False,
        with_geer=True,
        with_eval3d=True,
        camera_model=camera_model,
        radial_coeffs=radial_coeffs,
    )
    return rendered[0], alphas[0, ..., 0]


def gaussians_on_pixel_rays(camera_model, pixels, scale):
    """One Gaussian per pixel, sitting on that pixel's ray at depth ``DEPTH``."""
    num = len(pixels)
    means = torch.stack(
        [DEPTH * ray_direction(camera_model, px, py) for px, py in pixels]
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num, device=device).float()
    scales = torch.full((num, 3), scale, device=device, dtype=torch.float32)
    opacities = torch.ones((num,), device=device, dtype=torch.float32)
    colors = torch.ones((num, 3), device=device, dtype=torch.float32)
    return means, quats, scales, opacities, colors


# The pinhole case is a regression guard: it already behaved correctly and must
# keep doing so.  The fisheye case fails without the half open bounds fix.
@pytest.mark.parametrize("camera_model", ["fisheye", "pinhole"])
def test_last_column_and_row_are_drawn(camera_model):
    pixels = [(W - 1, H // 2), (W // 2, H - 1), (W - 1, H - 1), (0, 0)]
    scene = gaussians_on_pixel_rays(camera_model, pixels, scale=0.03)
    _, alpha = render(*scene, camera_model=camera_model)

    assert alpha[H // 2, W - 1].item() > 0.05, f"last column empty for {camera_model}"
    assert alpha[H - 1, W // 2].item() > 0.05, f"last row empty for {camera_model}"
    assert alpha[H - 1, W - 1].item() > 0.05, f"last corner empty for {camera_model}"
    assert alpha[0, 0].item() > 0.05, f"first pixel empty for {camera_model}"


@pytest.mark.parametrize("px,py", [(32, 24), (W - 2, H - 2), (5, H - 2), (W - 2, 3)])
def test_subpixel_gaussian_hits_its_own_pixel(px, py):
    """A Gaussian smaller than a pixel must survive and land on its own pixel."""
    scene = gaussians_on_pixel_rays("fisheye", [(px, py)], scale=0.002)
    _, alpha = render(*scene, camera_model="fisheye")

    assert alpha.max().item() > 0.05, f"sub pixel Gaussian vanished: {alpha.max()}"
    index = int(torch.argmax(alpha))
    assert (index // W, index % W) == (py, px)


def random_scene(num_gaussians=100, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.float32).to(device)

    offset = torch.tensor([0.0, 0.0, DEPTH], device=device)
    extent = torch.tensor([2.0, 2.0, 1.0], device=device)
    means = (rand(num_gaussians, 3) - 0.5) * extent + offset
    quats = torch.nn.functional.normalize(rand(num_gaussians, 4) - 0.5, dim=-1)
    scales = 0.02 + 0.08 * rand(num_gaussians, 3)
    opacities = 0.2 + 0.6 * rand(num_gaussians)
    colors = rand(num_gaussians, 3)
    return means, quats, scales, opacities, colors


def test_random_fisheye_scene_is_finite():
    rendered, alpha = render(*random_scene(), camera_model="fisheye")

    assert torch.isfinite(rendered).all()
    assert torch.isfinite(alpha).all()
    assert alpha.min().item() >= 0.0
    assert alpha.max().item() <= 1.0
    assert alpha.max().item() > 0.0
