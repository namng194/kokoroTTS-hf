"""Constrained-host profiles: the contribution core of this fork.

Upstream optimizes the full local Docker product (everything eager, GPU
first). This fork optimizes *deploy-anywhere*: free CPU Spaces, 8 GB VRAM
laptops, CPU-only servers. The mechanism is small and strictly opt-in:

- ``KOKOROTTS_PRELOAD``: ``all`` (default, today's behavior, unchanged),
  ``standard`` (eagerly prepare only the standard Kokoro family; German and
  Vietnamese families load on demand when an operator enables them), or
  ``lazy`` (prepare no voice packs up front; every voice loads on first use).
- ``KOKOROTTS_MAX_CHARS``: optional per-request text guard for shared/public
  deployments (free-tier abuse and OOM protection). Unset means unlimited,
  exactly as before.

This module is deliberately dependency-free (stdlib + ``catalog`` only) so
it can be unit-tested anywhere, including machines without torch.
"""

from __future__ import annotations

import os

from .catalog import STANDARD_MODEL_FAMILY

PRELOAD_ALL = "all"
PRELOAD_STANDARD = "standard"
PRELOAD_LAZY = "lazy"
PRELOAD_MODES = (PRELOAD_ALL, PRELOAD_STANDARD, PRELOAD_LAZY)


def is_space_env() -> bool:
    """True when running inside a Hugging Face Space (or forced)."""
    if os.getenv("SPACE_ID"):
        return True
    return os.getenv("KOKOROTTS_HF_SPACE", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def resolve_preload_mode() -> str:
    """Preload profile name; ``all`` preserves historical behavior."""
    mode = os.getenv("KOKOROTTS_PRELOAD", PRELOAD_ALL).strip().lower()
    if mode not in PRELOAD_MODES:
        raise ValueError(
            f"Unsupported KOKOROTTS_PRELOAD '{mode}'. "
            f"Use one of: {', '.join(PRELOAD_MODES)}."
        )
    return mode


def resolve_max_chars() -> int | None:
    """Per-request text limit, or None when unlimited (default)."""
    raw = os.getenv("KOKOROTTS_MAX_CHARS", "").strip()
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"KOKOROTTS_MAX_CHARS must be a positive integer, got '{raw}'."
        ) from exc
    if limit <= 0:
        raise ValueError(
            f"KOKOROTTS_MAX_CHARS must be a positive integer, got '{raw}'."
        )
    return limit


def check_text_length(text: str, *, limit: int | None = None) -> str:
    """Validate text against the length guard; returns the stripped text."""
    stripped = (text or "").strip()
    resolved = resolve_max_chars() if limit is None else limit
    if resolved is not None and len(stripped) > resolved:
        raise ValueError(
            f"Text is {len(stripped)} characters; this deployment "
            f"allows at most {resolved} per request."
        )
    return stripped


def apply_preload_defaults() -> str:
    """Seed served-family env defaults for lean-boot profiles.

    Only fills in ``KOKOROTTS_SERVED_MODEL_FAMILIES`` when the operator has
    not configured served voices/families by any means. Never touches the
    persisted settings file; that remains operator-owned via the System tab
    and the settings API.
    """
    mode = resolve_preload_mode()
    if mode == PRELOAD_STANDARD and not (
        os.getenv("KOKOROTTS_SERVED_MODEL_FAMILIES", "").strip()
        or os.getenv("KOKOROTTS_SERVED_VOICES", "").strip()
    ):
        os.environ.setdefault(
            "KOKOROTTS_SERVED_MODEL_FAMILIES", STANDARD_MODEL_FAMILY
        )
    return mode


def build_runtime_kwargs() -> dict[str, object]:
    """Keyword arguments for ``InferenceRuntime`` honoring the profile."""
    mode = apply_preload_defaults()
    return {"eager_voices": mode != PRELOAD_LAZY}


def profile_summary() -> dict[str, object]:
    """Small introspection payload for logs, the Gradio header, /tts/status."""
    return {
        "space_env": is_space_env(),
        "preload": resolve_preload_mode(),
        "max_chars": resolve_max_chars(),
        "served_model_families_env": os.getenv("KOKOROTTS_SERVED_MODEL_FAMILIES", ""),
        "served_voices_env": os.getenv("KOKOROTTS_SERVED_VOICES", ""),
    }
