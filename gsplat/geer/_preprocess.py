import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from ..cuda._wrapper import (
    FThetaCameraDistortionParameters,
    isect_tiles,
    isect_tiles_geer,
)
from .camera import get_camera_tanfov


def _antialias(
    scales: Tensor,
    opacities: Tensor,
    rasterize_mode: Literal["classic", "antialiased"],
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Inflate the 3D scales and compensate the mass, shared by both paths.

    3DGEER antialiasing inflates the 3D covariance rather than a projected 2D
    covariance. The inflated scales are used for the tile assignment and for
    the exact ray evaluation alike, and the determinant ratio preserves the
    Gaussian mass.

    Returns the (possibly inflated) scales, the opacities the tile assignment
    should see, and the compensations the caller multiplies into the opacities.
    """
    if rasterize_mode != "antialiased":
        return scales, opacities, None

    h_var = 1e-7
    antialiased_scales = torch.sqrt(scales.square() + h_var)
    compensations = torch.prod(scales / antialiased_scales, dim=-1)
    geer_opacities = opacities * compensations
    return antialiased_scales, geer_opacities, compensations[..., None, :]


def _prepare_geer(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"],
    rasterize_mode: Literal["classic", "antialiased"],
    packed: bool,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
) -> Tuple[
    Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]],
    Tuple[Tensor, Tensor, Tensor, Optional[Tensor]],
    Tensor,
]:
    """Prepare GEER metadata and tile intersections without a 2D projection."""
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]

    scales, geer_opacities, compensations = _antialias(
        scales, opacities, rasterize_mode
    )

    tanfovx, tanfovy = get_camera_tanfov(
        camera_model,
        Ks,
        width,
        height,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
    )

    tiles_per_gauss, isect_ids, flatten_ids, pbf_bounds = isect_tiles_geer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=geer_opacities,
        viewmats=viewmats,
        camera_model=camera_model,
        Ks=Ks,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=packed,
    )

    pbf_bounds = pbf_bounds.reshape(batch_dims + (C, N, 4))
    pbf_spans = torch.stack(
        (
            pbf_bounds[..., 1] - pbf_bounds[..., 0],
            pbf_bounds[..., 3] - pbf_bounds[..., 2],
        ),
        dim=-1,
    ).clamp_min_(0)
    radii = (pbf_spans + 1) // 2

    pbf_bounds_f = pbf_bounds.to(means)
    means2d = torch.stack(
        (
            0.5 * (pbf_bounds_f[..., 0] + pbf_bounds_f[..., 1]),
            0.5 * (pbf_bounds_f[..., 2] + pbf_bounds_f[..., 3]),
        ),
        dim=-1,
    )

    world_to_camera_z = viewmats[..., 2, :3]
    camera_z_offset = viewmats[..., 2, 3]
    depths = (
        world_to_camera_z @ means.transpose(-1, -2)
        + camera_z_offset[..., None]
    )
    conics = means.new_zeros(batch_dims + (C, N, 3))

    proj_results = radii, means2d, depths, conics, compensations
    intersections = tiles_per_gauss, isect_ids, flatten_ids, pbf_bounds
    return proj_results, intersections, scales


# Smallest image plane radius, relative to the distance, that still defines an
# azimuth. Below it the Gaussian centre sits on the optical axis.
_AXIS_EPS = 1e-9
# Keeps the pinhole derivative finite right at the horizon.
_PINHOLE_MAX_THETA = 0.5 * math.pi - 1e-3


def _fisheye_radius(theta: Tensor, radial_coeffs: Optional[Tensor]) -> Tensor:
    """Kannala Brandt image radius, in normalised image units.

    `r(theta) = theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9`, the
    very polynomial `OpenCVFisheyeCameraModel` builds. `theta` is `[..., C, N]`
    and `radial_coeffs` is `[..., C, 4]`.
    """
    series = torch.ones_like(theta)
    if radial_coeffs is not None:
        theta2 = theta * theta
        power = theta2
        for index in range(4):
            series = series + radial_coeffs[..., index][..., None] * power
            power = power * theta2
    return series * theta


def _fisheye_slope_bound(
    theta_max: Tensor, radial_coeffs: Optional[Tensor]
) -> Tensor:
    """Upper bound of `|dr/dtheta|` over `[0, theta_max]`.

    The derivative is `1 + 3 k1 theta^2 + 5 k2 theta^4 + 7 k3 theta^6 +
    9 k4 theta^8`, which is neither monotonic nor single signed once the
    coefficients are a fit rather than a lens, so sampling it at a few angles
    is not a bound. Replacing every coefficient by its magnitude makes each
    term non negative and increasing in `theta`, so evaluating that at the far
    end of the interval dominates `|dr/dtheta|` everywhere on it. For the
    equidistant model, where all coefficients vanish, this is exactly 1.
    """
    bound = torch.ones_like(theta_max)
    if radial_coeffs is not None:
        theta2 = theta_max * theta_max
        power = theta2
        for index in range(4):
            coeff = radial_coeffs[..., index][..., None].abs()
            bound = bound + (2 * index + 3) * coeff * power
            power = power * theta2
    return bound


def _project_centres(
    points: Tensor,
    distance: Tensor,
    Ks: Tensor,
    camera_model: Literal["pinhole", "fisheye"],
    radial_coeffs: Optional[Tensor],
    max_scales: Tensor,
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Project camera frame Gaussian centres and bound their pixel footprint.

    `points` is `[..., C, N, 3]` in the camera frame, `distance` its norm and
    `max_scales` is `[..., C, N]`. Returns the pixel centres `[..., C, N, 2]`,
    a conservative pixel radius `[..., C, N]` and a validity mask
    `[..., C, N]`.

    The polar angle comes from `atan2(rho, z)`, so it stays meaningful past 90
    degrees where `z` turns negative. The footprint bound treats the Gaussian
    as a ball of radius `3 sigma_max`: it subtends a cone of half angle
    `asin(3 sigma_max / d)` around the centre direction, which maps to a
    radial spread of `ang * max|dr/dtheta|` and an azimuthal spread of at most
    `asin(sin(ang) / sin(theta))`. Together they bound the distance from the
    projected centre to any point of the footprint.

    The pinhole branch keeps the behaviour of `_prepare_geer` and culls every
    centre at or behind the camera plane, so a Gaussian whose 3 sigma ball
    reaches across that plane while its centre sits behind it is dropped even
    though its tail would still be visible. Only the fisheye branch is meant
    to work past 90 degrees.
    """
    # A disk at the image centre with half the diagonal as its radius covers
    # the whole image; one full diagonal covers it from anywhere nearby and
    # caps a Kannala Brandt polynomial that diverges past its fitted range.
    half_diagonal = 0.5 * math.hypot(float(width), float(height))
    radius_cap = 2.0 * half_diagonal

    fx = Ks[..., 0, 0][..., None]  # [..., C, 1]
    fy = Ks[..., 1, 1][..., None]
    cx = Ks[..., 0, 2][..., None]
    cy = Ks[..., 1, 2][..., None]
    focal = torch.maximum(fx, fy)

    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    rho = torch.sqrt(x * x + y * y)

    safe_rho = rho.clamp_min(torch.finfo(points.dtype).tiny)
    on_axis = rho <= _AXIS_EPS * distance.clamp_min(torch.finfo(points.dtype).tiny)
    direction_x = torch.where(on_axis, torch.zeros_like(x), x / safe_rho)
    direction_y = torch.where(on_axis, torch.zeros_like(y), y / safe_rho)

    # Angular radius of the 3 sigma ball. A camera inside that ball sees the
    # Gaussian in every direction, which the full image radius below covers.
    sin_ang = (3.0 * max_scales) / distance.clamp_min(torch.finfo(points.dtype).tiny)
    engulfing = sin_ang >= 1.0
    ang = torch.asin(sin_ang.clamp(max=1.0))

    if camera_model == "fisheye":
        theta = torch.atan2(rho, z)
        radius_norm = _fisheye_radius(theta, radial_coeffs)
        max_slope = _fisheye_slope_bound(
            (theta + ang).clamp_max(math.pi), radial_coeffs
        )
        # A centre exactly behind the camera has no azimuth, so its projection
        # is undefined. The ring it would draw is left out on purpose.
        valid = ~(on_axis & (z < 0.0))
    else:
        theta = torch.atan2(rho, z.clamp_min(torch.finfo(points.dtype).tiny))
        theta = theta.clamp(max=_PINHOLE_MAX_THETA)
        theta_hi = (theta + ang).clamp(max=_PINHOLE_MAX_THETA)
        radius_norm = torch.tan(theta)
        # 1 / cos^2 grows with theta, so the far end of the interval bounds it.
        max_slope = 1.0 / torch.cos(theta_hi).square()
        valid = z > 0.0

    radial_spread = ang * max_slope
    radius_max = radius_norm + radial_spread
    # Azimuthal half span of the cone around a direction at polar angle theta.
    sin_theta = torch.sin(theta).clamp_min(torch.finfo(points.dtype).tiny)
    sin_ratio = (torch.sin(ang) / sin_theta).clamp(max=1.0)
    azimuth_span = torch.where(
        (ang >= theta) | (ang >= math.pi - theta) | engulfing,
        torch.full_like(ang, math.pi),
        torch.asin(sin_ratio),
    )
    # A footprint point at image radius R and azimuth offset `delta` sits at
    # squared distance (R - r0)^2 + 2 r0 R (1 - cos delta) from the projected
    # centre, which the radial and the azimuthal spread bound term by term.
    # Every footprint point also sits at image radius at most `radius_max`, so
    # going through the principal point is the better bound whenever the cone
    # wraps a large span of azimuths.
    through_centre = radius_norm + radius_max
    diagonal = torch.sqrt(
        radial_spread.square() + (radius_max * azimuth_span).square()
    )
    radius = (focal * torch.minimum(diagonal, through_centre)).clamp(
        max=radius_cap
    )

    means2d = torch.stack(
        (
            cx + fx * radius_norm * direction_x,
            cy + fy * radius_norm * direction_y,
        ),
        dim=-1,
    )
    valid = valid & torch.isfinite(means2d).all(dim=-1) & torch.isfinite(radius)

    # A camera inside the 3 sigma ball sees the Gaussian in every direction,
    # so cover the whole image rather than a disk around a meaningless centre.
    everywhere = torch.stack(
        (
            torch.full_like(radius, 0.5 * float(width)),
            torch.full_like(radius, 0.5 * float(height)),
        ),
        dim=-1,
    )
    means2d = torch.where(engulfing[..., None], everywhere, means2d)
    radius = torch.where(
        engulfing, torch.full_like(radius, half_diagonal), radius
    )
    valid = valid | engulfing
    return means2d, radius, valid


def _prepare_geer_projection(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"],
    rasterize_mode: Literal["classic", "antialiased"],
    packed: bool,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
) -> Tuple[
    Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]],
    Tuple[Tensor, Tensor, Tensor, Optional[Tensor]],
    Tensor,
]:
    """Assign tiles by projecting the Gaussian centres with the camera model.

    The default 3DGEER path builds a perspective bounding frustum in tangent
    space, which needs the Gaussian centre in front of the camera plane and a
    finite tangent of the field of view. This path drops both: it projects the
    centres through the real camera model, where `atan2(rho, z)` keeps working
    past 90 degrees, bounds the pixel footprint from the 3 sigma ball, and
    hands the result to the stock `isect_tiles`. The exact per ray evaluation
    in the rasterizer is unchanged; only the set of Gaussians a tile looks at
    is built differently.

    Differences to `_prepare_geer` that a caller can observe:

    * `depths` is the euclidean distance to the camera centre, not the camera
      frame z. Beyond 90 degrees z is negative and useless both as a sort key
      and as a depth value, so "D" and "ED" renders carry distances here. It
      is differentiable in `means` and `viewmats` like the camera frame z of
      `_prepare_geer`; only the tile assignment runs detached.
    * `near_plane` and `far_plane` cut on that distance as well.
    * The per Gaussian pixel bounds (`pbf_bounds`) are gone, so a tile tests
      every Gaussian that reaches it. Tiles are the only culling left.

    Like `_prepare_geer` this renders one image at a time. The maths here would
    take any number of cameras, but `isect_tiles` writes a tile range table
    that is sized for a single image while it indexes it with the image id
    folded in (`identify_tile_ranges` in `IntersectTile.cu`, called from
    `Intersect.cpp`), so more than one image corrupts memory next to that
    table. That table is dead output, nothing reads it, but fixing it belongs
    to the shared tile assignment and not here.

    Known limits: only `pinhole` and `fisheye` are supported, and only the
    fisheye branch works past 90 degrees, since the pinhole branch still culls
    every centre at or behind the camera plane. A Gaussian whose centre sits
    exactly on the optical axis behind the camera is dropped. The footprint of
    a Gaussian very close to 180 degrees is a ring that the disk around its
    projected centre only bounds up to the image diagonal cap. A Kannala
    Brandt fit (a UCM handed over as a fisheye, say) is only valid up to the
    angle it was fitted on; beyond that the polynomial diverges and the radius
    runs into the image diagonal cap.
    """
    if camera_model not in ("pinhole", "fisheye"):
        raise ValueError(
            "geer_tile_mode='projection' supports the 'pinhole' and 'fisheye' "
            f"camera models only, got '{camera_model}'"
        )
    if tangential_coeffs is not None or thin_prism_coeffs is not None:
        raise ValueError(
            "geer_tile_mode='projection' does not model tangential or thin "
            "prism distortion"
        )
    if camera_model == "pinhole" and radial_coeffs is not None:
        raise ValueError(
            "geer_tile_mode='projection' does not model radial distortion for "
            "the 'pinhole' camera model"
        )
    assert not packed, "geer_tile_mode='projection' does not support packed mode"

    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    if math.prod(batch_dims) * C != 1:
        raise ValueError(
            "geer_tile_mode='projection' renders one image at a time, got "
            f"{math.prod(batch_dims) * C}; see the note on `isect_tiles` above"
        )

    scales, _, compensations = _antialias(scales, opacities, rasterize_mode)

    # The distance doubles as the depth that the "D" and "ED" render modes
    # draw, so it is built with the graph attached, exactly as `_prepare_geer`
    # builds its camera frame z. Everything the tile assignment needs is taken
    # from the detached copies below.
    rotation = viewmats[..., :3, :3].transpose(-1, -2)  # [..., C, 3, 3]
    translation = viewmats[..., :3, 3][..., None, :]  # [..., C, 1, 3]
    points = means[..., None, :, :] @ rotation + translation  # [..., C, N, 3]
    depths = points.norm(dim=-1)  # [..., C, N]

    with torch.no_grad():
        points_detached = points.detach()
        distance = depths.detach()

        max_scales = scales.detach().amax(dim=-1)[..., None, :]  # [..., 1, N]
        max_scales = max_scales.expand(batch_dims + (C, N))

        means2d, radius, valid = _project_centres(
            points_detached,
            distance,
            Ks,
            camera_model,
            radial_coeffs,
            max_scales,
            width,
            height,
        )

        valid = (
            valid
            & (distance > near_plane)
            & (distance < far_plane)
            & (radius > radius_clip)
        )
        radii_1d = torch.where(
            valid, torch.ceil(radius), torch.zeros_like(radius)
        ).to(torch.int32)
        radii = torch.stack((radii_1d, radii_1d), dim=-1)  # [..., C, N, 2]
        means2d = torch.where(valid[..., None], means2d, torch.zeros_like(means2d))
        # A culled Gaussian reaches no tile, but a non finite sort key would
        # still scramble the order of the ones that do.
        sort_keys = torch.where(
            torch.isfinite(distance), distance, torch.zeros_like(distance)
        )

        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d.contiguous(),
            radii.contiguous(),
            sort_keys.contiguous(),
            tile_size,
            tile_width,
            tile_height,
            packed=False,
        )

    conics = means.new_zeros(batch_dims + (C, N, 3))
    proj_results = radii, means2d, depths, conics, compensations
    intersections = tiles_per_gauss, isect_ids, flatten_ids, None
    return proj_results, intersections, scales
