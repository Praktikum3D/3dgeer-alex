"""Tests for the optional 3DGEER densification proxy `meta["geer_gradient"]`.

The proxy is the densification metric of the 3DGEER paper.  Upstream allocated
it on every `with_geer` call, which costs one [..., C, N, 3] tensor and three
atomic adds per Gaussian and pixel in the backward pass, even though only
`DefaultStrategy` ever reads it.  It is now behind `with_geer_gradient`.

The checks here are:

* without the flag the proxy is `None` and forward and backward still run,
* with the flag the proxy is a tensor whose gradient is populated and nonzero,
* every other gradient is the same with and without the flag,
* the flag is rejected without `with_geer`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_ray_gradients as reference_tests  # noqa: E402

device = torch.device("cuda:0")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")


W, H = reference_tests.W, reference_tests.H
CAMERA_MODELS = ["pinhole", "fisheye"]


def _cotangent(seed=3):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.rand(1, H, W, 3, generator=generator, dtype=torch.float32) - 0.5
    return values.to(device)


def render(camera_model, with_geer_gradient, with_rays=False, seed=0):
    """One `rasterization` call through the 3DGEER path, differentiable."""
    from gsplat import rasterization

    scene_tensors = [
        tensor.clone().requires_grad_(True)
        for tensor in reference_tests.scene(seed=seed)
    ]
    means, quats, scales, opacities, colors = scene_tensors
    viewmat = reference_tests._viewmat()

    rays_o, rays_d = None, None
    if with_rays:
        rays_o, rays_d = reference_tests.pixel_rays(camera_model, viewmat)
        rays_o = rays_o[None].requires_grad_(True)
        rays_d = rays_d[None].requires_grad_(True)
        scene_tensors += [rays_o, rays_d]

    render_colors, _, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmat,
        reference_tests._Ks(),
        W,
        H,
        packed=False,
        with_geer=True,
        with_geer_gradient=with_geer_gradient,
        with_eval3d=True,
        camera_model=camera_model,
        radial_coeffs=reference_tests._radial_coeffs(camera_model),
        rays_o=rays_o,
        rays_d=rays_d,
    )
    return render_colors, meta, scene_tensors


def backward(render_colors, meta, scene_tensors):
    """Gradients of a fixed linear loss, with the proxy retained if present."""
    proxy = meta["geer_gradient"]
    if proxy is not None:
        proxy.retain_grad()
    (render_colors * _cotangent()).sum().backward()
    return [tensor.grad for tensor in scene_tensors]


@pytest.mark.parametrize("camera_model", CAMERA_MODELS)
def test_proxy_is_absent_by_default(camera_model):
    render_colors, meta, scene_tensors = render(camera_model, with_geer_gradient=False)
    assert "geer_gradient" in meta
    assert meta["geer_gradient"] is None

    grads = backward(render_colors, meta, scene_tensors)
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().max() > 0 for grad in grads)


@pytest.mark.parametrize("camera_model", CAMERA_MODELS)
def test_proxy_is_populated_with_the_flag(camera_model):
    render_colors, meta, scene_tensors = render(camera_model, with_geer_gradient=True)
    proxy = meta["geer_gradient"]
    assert torch.is_tensor(proxy)
    num_gaussians = scene_tensors[0].shape[0]
    assert proxy.shape == (1, num_gaussians, 3)
    assert proxy.requires_grad

    backward(render_colors, meta, scene_tensors)
    assert proxy.grad is not None
    assert torch.isfinite(proxy.grad).all()
    assert proxy.grad.abs().max() > 0


@pytest.mark.parametrize("camera_model", CAMERA_MODELS)
@pytest.mark.parametrize("with_rays", [False, True])
def test_the_flag_leaves_every_other_gradient_alone(camera_model, with_rays):
    off = backward(*render(camera_model, False, with_rays=with_rays))
    on = backward(*render(camera_model, True, with_rays=with_rays))

    assert len(off) == len(on)
    for grad_off, grad_on in zip(off, on):
        # The backward pass accumulates with atomics, so the summation order
        # differs between the two runs and the results are only close.
        scale = max(grad_off.abs().max().item(), 1e-6)
        torch.testing.assert_close(
            grad_off, grad_on, rtol=1e-5, atol=1e-5 * scale, equal_nan=False
        )


@pytest.mark.parametrize("camera_model", CAMERA_MODELS)
def test_renders_are_identical_with_and_without_the_flag(camera_model):
    colors_off, _, _ = render(camera_model, with_geer_gradient=False)
    colors_on, _, _ = render(camera_model, with_geer_gradient=True)
    torch.testing.assert_close(colors_off, colors_on, rtol=0.0, atol=0.0)


def test_the_proxy_is_absent_without_grad():
    with torch.no_grad():
        _, meta, _ = render("pinhole", with_geer_gradient=True)
    assert meta["geer_gradient"] is None


def test_the_flag_needs_geer():
    from gsplat import rasterization

    means, quats, scales, opacities, colors = reference_tests.scene()
    with pytest.raises(AssertionError, match="geer gradient"):
        rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            reference_tests._viewmat(),
            reference_tests._Ks(),
            W,
            H,
            packed=False,
            with_geer=False,
            with_geer_gradient=True,
        )
