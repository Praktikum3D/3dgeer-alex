# `3dgeer-alex`

A fork of [`boschresearch/3dgeer`](https://github.com/boschresearch/3dgeer) (`gsplat-geer`), which extends
[`gsplat`](https://github.com/nerfstudio-project/gsplat) 1.5.3 from
[Nerfstudio](https://docs.nerf.studio/) with exact ray-Gaussian rendering ("3DGEER").

This fork is maintained for the TUM CAMP 3D Computer Vision Praktikum, where 3DGEER is integrated into
the E-RayZer pipeline. It is trimmed down to the library itself: the example trainers, viewers,
benchmarks, profiling scripts, the Sphinx documentation and the DriveStudio integration of the
upstream repository have been removed. Only `gsplat/` and the tests that cover it remain.

## Install

```bash
git clone --recurse-submodules https://github.com/Praktikum3D/3dgeer-alex.git
cd 3dgeer-alex
pip install --no-build-isolation -e .
```

Or directly from GitHub:

```bash
pip install --no-build-isolation git+https://github.com/Praktikum3D/3dgeer-alex.git
```

The header-only [glm](https://github.com/g-truc/glm) library is a pinned git submodule under
`gsplat/cuda/csrc/third_party/glm` and is required at compile time, so always clone with
`--recurse-submodules`. Building the CUDA extension needs a CUDA toolkit matching the installed
PyTorch; set `TORCH_CUDA_ARCH_LIST` to your GPU architecture to keep compile times down.

## Usage

3DGEER is enabled through the usual `gsplat` entry point by calling
`rasterization(..., packed=False, with_geer=True, with_eval3d=True)`:

```python
from gsplat import rasterization

renders, alphas, meta = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height,
    packed=False,          # required: the 3DGEER path does not support packed mode
    sparse_grad=False,     # required: the 3DGEER path does not support sparse gradients
    with_geer=True,
    with_eval3d=True,
    camera_model="pinhole",  # or "fisheye" / "ftheta"
)
```

Notes from upstream that still apply:

- `with_geer=True` only works with `with_eval3d=True`.
- `with_geer=True` requires `packed=False` and `sparse_grad=False`; note that `packed` defaults to `True`, so it has to be passed explicitly.
- `with_geer` only renders one image plane at a time.
- Training is most stable with the default strategy.
- To train/render pinhole camera with distortion, set the distortion parameters to `radial_coeffs`, `tangential_coeffs`, `thin_prism_coeffs`.
- To train/render fisheye camera with distortion, set the distortion parameters to `radial_coeffs` and set `camera_model="fisheye"`.
- To train/render an F-theta camera, set `ftheta_coeffs` and `camera_model="ftheta"`.

These are consistent with `gsplat`'s 3DGUT implementation (`with_ut`).

One deviation from upstream: the auxiliary densification tensor `meta["geer_gradient"]` of the
3DGEER paper is only allocated when `rasterization(..., with_geer_gradient=True)` is passed, and is
`None` otherwise. Upstream allocated it whenever autograd was enabled, which costs one `[..., C, N, 3]` tensor plus three
atomic adds per Gaussian and pixel in the backward pass even though only `DefaultStrategy` ever
reads it. Pass the flag when densifying with that metric.

## Tests

The tests need a CUDA GPU.

```bash
pytest tests/
```

## License

`3dgeer-alex` is released under the AGPL-3.0 License, see [LICENSE.md](./LICENSE.md). It is built upon
`gsplat` (Apache-2.0, Copyright 2025 Nerfstudio Team); the original license and copyright notice are
kept in [gsplat-license.txt](./gsplat-license.txt).

## Origin and credits

- Upstream fork source: [boschresearch/3dgeer](https://github.com/boschresearch/3dgeer), by the Bosch Center for AI.
- Underlying library: [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3, by the Nerfstudio Team.
- 3DGEER paper: Zixun Huang, Cho-Ying Wu, Yuliang Guo, Xinyu Huang, Liu Ren, *3DGEER: 3D Gaussian
  Rendering Made Exact and Efficient for Generic Cameras*, [arXiv:2505.24053](https://arxiv.org/abs/2505.24053).
  See [CITATION.bib](./CITATION.bib).
