"""Voice blending: mix two served voices into a new synthetic speaker.

This is this fork's own inference feature — upstream serves fixed voices
only. A Kokoro voice pack is a tensor of per-token reference vectors with
shape ``(tokens, 256)``; blending is a weighted average of two packs with
identical shape. Endpoints are exact: weight 0.0 reproduces voice A and
weight 1.0 reproduces voice B.

Only voices that share one model family can blend (different families use
different weight spaces); the API enforces this with HTTP 400.
"""

from __future__ import annotations

import torch


def parse_blend_weight(value: object) -> float:
    """Coerce a blend weight to float and require the 0..1 range."""
    try:
        weight = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Blend weight must be a number in 0..1, got {value!r}."
        ) from exc
    if not 0.0 <= weight <= 1.0:
        raise ValueError(
            f"Blend weight must be in 0..1, got {weight}."
        )
    return weight


def check_blend_compatible(pack_a: torch.Tensor, pack_b: torch.Tensor) -> None:
    """Reject pack pairs that cannot mix (different shapes/dtypes)."""
    if pack_a.shape != pack_b.shape:
        raise ValueError(
            f"Voices have incompatible packs {tuple(pack_a.shape)} vs "
            f"{tuple(pack_b.shape)} and cannot blend. Pick two voices from "
            "the same model family."
        )
    if pack_a.dtype != pack_b.dtype:
        raise ValueError(
            f"Voices have incompatible pack dtypes {pack_a.dtype} vs "
            f"{pack_b.dtype} and cannot blend."
        )


def blend_packs(
    pack_a: torch.Tensor, pack_b: torch.Tensor, weight_b: float = 0.5
) -> torch.Tensor:
    """Return the blended pack: ``(1-w)*A + w*B`` element-wise.

    Raises :class:`ValueError` on out-of-range weights or incompatible packs.
    """
    weight = parse_blend_weight(weight_b)
    check_blend_compatible(pack_a, pack_b)
    return (1.0 - weight) * pack_a + weight * pack_b
