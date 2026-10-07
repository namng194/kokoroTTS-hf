"""Checkpoint sourcing policy.

By default KokoroTTS may download missing weights from the Hugging Face
Hub on first use. When KOKOROTTS_BUNDLED_CHECKPOINTS=1 (the HF Space),
every weight/config/voice lookup is forced local-only: a missing bundled
file fails fast instead of reaching the network.
"""

from __future__ import annotations

import os

FLAG = "KOKOROTTS_BUNDLED_CHECKPOINTS"


def bundled_only() -> bool:
    return os.getenv(FLAG, "0").strip().lower() in {"1", "true", "yes"}
