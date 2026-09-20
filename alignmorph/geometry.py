"""Coordinate utilities shared by correspondence and latent transport.

Point maps in this package always use pixel-centre coordinates in ``(x, y)``
order.  A map ``A -> B`` is defined on the A lattice and stores positions on
the B lattice.  It is therefore not directly a ``grid_sample`` pullback grid.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def identity_map(
    height: int,
    width: int,
    *,
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an ``(H,W,2)`` identity point map in ``(x,y)`` order."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1)


def pixel_to_normalized(
    points: torch.Tensor,
    height: int,
    width: int,
    *,
    align_corners: bool = False,
) -> torch.Tensor:
    """Convert pixel-centre coordinates to PyTorch sampling coordinates."""

    if points.shape[-1] != 2:
        raise ValueError("points must end in an (x,y) coordinate dimension")
    if align_corners:
        x = 2.0 * points[..., 0] / float(max(width - 1, 1)) - 1.0
        y = 2.0 * points[..., 1] / float(max(height - 1, 1)) - 1.0
    else:
        x = 2.0 * (points[..., 0] + 0.5) / float(width) - 1.0
        y = 2.0 * (points[..., 1] + 0.5) / float(height) - 1.0
    return torch.stack((x, y), dim=-1)


def sample_field(
    field: torch.Tensor,
    points: torch.Tensor,
    *,
    padding_mode: str = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    """Sample an ``(H,W,C)`` field at pixel-space ``(Hq,Wq,2)`` points."""

    if field.ndim != 3 or points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("field must be (H,W,C) and points must be (Hq,Wq,2)")
    grid = pixel_to_normalized(
        points, field.shape[0], field.shape[1], align_corners=align_corners
    ).unsqueeze(0)
    sampled = F.grid_sample(
        field.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return sampled.squeeze(0).permute(1, 2, 0)


def sample_scalar(
    field: torch.Tensor,
    points: torch.Tensor,
    *,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Sample an ``(H,W)`` scalar field at pixel-space points."""

    if field.ndim != 2:
        raise ValueError("field must be (H,W)")
    return sample_field(
        field.unsqueeze(-1),
        points,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )[..., 0]


def resize_point_map(
    point_map: torch.Tensor,
    output_hw: Tuple[int, int],
    *,
    target_input_hw: Optional[Tuple[int, int]] = None,
    target_output_hw: Optional[Tuple[int, int]] = None,
    align_corners: bool = False,
) -> torch.Tensor:
    """Resize a point map by interpolating and rescaling displacement (Eq. 5).

    ``point_map`` is defined on its own source grid.  By default source and
    target grids have the same size.  The explicit target sizes also support
    rectangular or differently-sized feature lattices.
    """

    if point_map.ndim != 3 or point_map.shape[-1] != 2:
        raise ValueError("point_map must have shape (H,W,2)")
    source_input_hw = tuple(int(v) for v in point_map.shape[:2])
    source_output_hw = tuple(int(v) for v in output_hw)
    target_input_hw = (
        source_input_hw if target_input_hw is None else tuple(target_input_hw)
    )
    target_output_hw = (
        source_output_hw if target_output_hw is None else tuple(target_output_hw)
    )
    if min(*source_output_hw, *target_input_hw, *target_output_hw) <= 0:
        raise ValueError("all grid dimensions must be positive")

    source_identity = identity_map(
        *source_input_hw, device=point_map.device, dtype=point_map.dtype
    )
    if align_corners:
        sx_in = float(max(target_input_hw[1] - 1, 1)) / float(
            max(source_input_hw[1] - 1, 1)
        )
        sy_in = float(max(target_input_hw[0] - 1, 1)) / float(
            max(source_input_hw[0] - 1, 1)
        )
        identity_in_target = torch.stack(
            (source_identity[..., 0] * sx_in, source_identity[..., 1] * sy_in),
            dim=-1,
        )
    else:
        identity_in_target = torch.stack(
            (
                (source_identity[..., 0] + 0.5)
                * float(target_input_hw[1])
                / float(source_input_hw[1])
                - 0.5,
                (source_identity[..., 1] + 0.5)
                * float(target_input_hw[0])
                / float(source_input_hw[0])
                - 0.5,
            ),
            dim=-1,
        )
    displacement = torch.nan_to_num(point_map - identity_in_target)
    displacement = (
        F.interpolate(
            displacement.permute(2, 0, 1).unsqueeze(0),
            size=source_output_hw,
            mode="bilinear",
            align_corners=align_corners,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )

    if align_corners:
        sx = float(max(target_output_hw[1] - 1, 1)) / float(
            max(target_input_hw[1] - 1, 1)
        )
        sy = float(max(target_output_hw[0] - 1, 1)) / float(
            max(target_input_hw[0] - 1, 1)
        )
    else:
        sx = float(target_output_hw[1]) / float(target_input_hw[1])
        sy = float(target_output_hw[0]) / float(target_input_hw[0])
    displacement = displacement * torch.tensor(
        (sx, sy), device=point_map.device, dtype=point_map.dtype
    )

    output_identity = identity_map(
        *source_output_hw, device=point_map.device, dtype=point_map.dtype
    )
    if source_output_hw != target_output_hw:
        if align_corners:
            output_identity = torch.stack(
                (
                    output_identity[..., 0]
                    * float(max(target_output_hw[1] - 1, 1))
                    / float(max(source_output_hw[1] - 1, 1)),
                    output_identity[..., 1]
                    * float(max(target_output_hw[0] - 1, 1))
                    / float(max(source_output_hw[0] - 1, 1)),
                ),
                dim=-1,
            )
        else:
            output_identity = torch.stack(
                (
                    (output_identity[..., 0] + 0.5)
                    * float(target_output_hw[1])
                    / float(source_output_hw[1])
                    - 0.5,
                    (output_identity[..., 1] + 0.5)
                    * float(target_output_hw[0])
                    / float(source_output_hw[0])
                    - 0.5,
                ),
                dim=-1,
            )
    return output_identity + displacement


def resize_scalar(
    field: torch.Tensor,
    output_hw: Tuple[int, int],
    *,
    align_corners: bool = False,
) -> torch.Tensor:
    if field.ndim != 2:
        raise ValueError("field must be (H,W)")
    return (
        F.interpolate(
            field.unsqueeze(0).unsqueeze(0),
            size=output_hw,
            mode="bilinear",
            align_corners=align_corners,
        )
        .squeeze(0)
        .squeeze(0)
    )
