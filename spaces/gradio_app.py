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

from kokorotts.catalog import (
    LANGUAGE_CHOICES,
    MODEL_FAMILY_CHOICES,
    voice_ids,
    voice_label,
    voice_language,
)
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


ALL_FAMILIES = list(MODEL_FAMILY_CHOICES.keys())


def _synthesize(text: str, voice: str, speed: float, voice_blend):
    runtime = get_runtime()
    try:
        return runtime.synthesize(
            text=text, voice=voice, speed=speed,
            device=_inference_device(), input_type="text",
            voice_blend=voice_blend,
        )
    except ValueError as exc:
        if "not served" not in str(exc):
            raise
        # Lean-boot deployments only serve the standard family: enable
        # everything (downloads German/Vietnamese packs on first use) once.
        runtime.set_served_model_families(ALL_FAMILIES)
        return runtime.synthesize(
            text=text, voice=voice, speed=speed,
            device=_inference_device(), input_type="text",
            voice_blend=voice_blend,
        )


def generate(text: str, mode: str, voice: str, voice_b: str, speed: float, mix: float):
    try:
        text = check_text_length(text)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not text:
        raise ValueError("Please enter some text first.")
    speed = max(0.25, min(4.0, float(speed or 1.0)))
    voice_blend = None
    if mode == "Blend":
        from kokorotts.catalog import voice_model_family
        from kokorotts.voice_blend import parse_blend_weight

        if voice_b == voice:
            raise ValueError("Pick two different voices to blend.")
        mix = parse_blend_weight(mix)
        if voice_model_family(voice_b) != voice_model_family(voice):
            raise ValueError("Blended voices must share one model family.")
        voice_blend = (voice_b, mix)
    runtime = get_runtime()
    result = _synthesize(text, voice, speed, voice_blend)
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
            mode = gr.Radio(
                label="Mode", choices=["Single", "Blend"], value="Single"
            )
            voice = gr.Dropdown(
                label="Voice A (70, grouped by language)",
                choices=voice_choices(),
                value=DEFAULT_VOICE,
            )
        with gr.Row(visible=False) as blend_row:
            voice_b = gr.Dropdown(
                label="Voice B (must share A's model family)",
                choices=voice_choices(),
                value="af_bella",
            )
            mix = gr.Slider(
                label="Mix: weight of voice B (0 = pure A, 1 = pure B)",
                minimum=0.0, maximum=1.0, step=0.05, value=0.5,
            )
        with gr.Row():
            speed = gr.Slider(
                label="Speed", minimum=0.25, maximum=4.0, step=0.05, value=1.0
            )
        gr.Markdown(
            "_Blend_ mixes two voices into a new synthetic speaker — "
            "a KokoroTTS-HF exclusive."
        )
        btn = gr.Button("Generate", variant="primary")
        audio = gr.Audio(label="Output (24 kHz)")

        def _toggle_blend(selected: str):
            return gr.Row(visible=(selected == "Blend"))

        mode.change(fn=_toggle_blend, inputs=mode, outputs=blend_row)
        btn.click(
            fn=generate,
            inputs=[text, mode, voice, voice_b, speed, mix],
            outputs=audio,
            api_name="generate",
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
