"""Full-catalogue Gradio demo for Hugging Face Spaces (free CPU tier).

This is the *demo* Space flavour (``sdk: gradio``). Unlike a minimal teaser,
it serves the complete 70-voice catalogue grouped by language and honors the
constrained-host profiles from ``kokorotts.space``:

- ``KOKOROTTS_PRELOAD``: ``all`` (default) | ``standard`` | ``lazy``
- ``KOKOROTTS_MAX_CHARS``: per-request text guard for public deployments

The recommended production flavour remains the Docker Space (``Dockerfile.hf``)
with the full browser workspace + OpenAI-compatible API; see
``docs/HF_SPACES.md``. Both flavours share one inference stack, so a voice
that works here works identically there.

Run locally (CPU, no GPU touched):
    pip install -r spaces/requirements-gradio.txt
    python spaces/gradio_app.py
"""

from __future__ import annotations

import os
import threading

import spaces

from kokorotts.catalog import LANGUAGE_CHOICES, voice_ids, voice_label, voice_language
from kokorotts.space import (
    build_runtime_kwargs,
    check_text_length,
    profile_summary,
)

SAMPLE_RATE = 24000
DEFAULT_VOICE = os.getenv("KOKOROTTS_HF_DEFAULT_VOICE", "af_heart")
DEFAULT_TEXT = (
    "Hello from KokoroTTS-HF on Hugging Face Spaces. "
    "Type any text, pick any of the 70 voices, and press Generate."
)

_runtime = None
_runtime_lock = threading.Lock()


def voice_choices() -> list[tuple[str, str]]:
    """All voices as (label, id) pairs grouped by language for Gradio."""
    grouped: list[tuple[str, str]] = []
    for code in LANGUAGE_CHOICES:
        for voice_id in voice_ids():
            if voice_language(voice_id) == code:
                grouped.append(
                    (f"{LANGUAGE_CHOICES[code]} — {voice_label(voice_id)}", voice_id)
                )
    return grouped


def get_runtime():
    """Lazily build the inference runtime once (thread-safe)."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        from kokorotts.runtime import InferenceRuntime
        from kokorotts.settings import RuntimeSettingsStore

        _runtime = InferenceRuntime(
            settings=RuntimeSettingsStore(), **build_runtime_kwargs()
        )
        return _runtime


def _inference_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@spaces.GPU(duration=120)
def generate(text: str, voice: str, speed: float):
    try:
        text = check_text_length(text)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not text:
        raise ValueError("Please enter some text first.")
    speed = max(0.25, min(4.0, float(speed or 1.0)))
    runtime = get_runtime()
    result = runtime.synthesize(
        text=text, voice=voice, speed=speed,
        device=_inference_device(), input_type="text",
    )
    if result is None:
        raise RuntimeError("Synthesis returned no audio.")
    return SAMPLE_RATE, result.audio


def build_demo():
    import gradio as gr

    summary = profile_summary()
    limit = summary["max_chars"]
    guard_line = (
        f"Max {limit} characters per request on this deployment."
        if limit else "No per-request character limit on this deployment."
    )

    with gr.Blocks(title="KokoroTTS-HF (70 voices, CPU)") as demo:
        gr.Markdown(
            "# KokoroTTS-HF community demo\n"
            "Full 70-voice catalogue on CPU. For the browser workspace + "
            "OpenAI-compatible API, use the Docker Space flavour "
            "(`docs/HF_SPACES.md`).\n\n"
            f"{guard_line} Profile: `{summary['preload']}`.\n\n"
            "Based on Hangry Labs KokoroTTS (Apache-2.0) and hexgrad Kokoro."
        )
        with gr.Row():
            text = gr.Textbox(label="Text", value=DEFAULT_TEXT, lines=5, max_lines=12)
        with gr.Row():
            voice = gr.Dropdown(
                label="Voice (70, grouped by language)",
                choices=voice_choices(),
                value=DEFAULT_VOICE,
            )
            speed = gr.Slider(
                label="Speed", minimum=0.25, maximum=4.0, step=0.05, value=1.0
            )
        btn = gr.Button("Generate", variant="primary")
        audio = gr.Audio(label="Output (24 kHz)")
        btn.click(fn=generate, inputs=[text, voice, speed], outputs=audio)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
