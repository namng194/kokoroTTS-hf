"""Lightweight Gradio demo for Hugging Face Spaces (free CPU tier).

This is the *alternative* Space flavour: `sdk: gradio`, no Docker build step,
small cold start. The recommended production flavour is the Docker Space
(see Dockerfile.hf + docs/HF_SPACES.md) which serves the full browser UI and
both HTTP APIs. This Gradio app exists so the project is one-click deployable
on a free CPU Space.

Design notes for constrained hosts:
- CPU only (`KOKOROTTS_DEVICE=cpu` default); Kokoro-82M runs fine on CPU,
  just slower than GPU. First request downloads weights (~300MB) then caches
  under HF_HOME (/data on Spaces).
- Lazy singleton runtime: nothing loads at import time, so the Space boots
  fast and only pays inference cost on first generation.
- Small default voice list keeps the dropdown usable; any voice id accepted
  via the API flavour remains available in the Docker Space.
- Experimental SSML is intentionally NOT exposed here; plain text only.

Run locally (CPU, no GPU touched):
    pip install -r spaces/requirements-gradio.txt
    python spaces/gradio_app.py
"""

from __future__ import annotations

import os
import threading

SAMPLE_RATE = 24000
DEFAULT_VOICE = os.getenv("KOKOROTTS_HF_DEFAULT_VOICE", "af_heart")
DEFAULT_TEXT = (
    "Hello from KokoroTTS on Hugging Face Spaces. "
    "Type any text, pick a voice, and press Generate."
)

# A compact, multilingual starter set. Full 70-voice catalogue lives in
# kokorotts/catalog.py and in the Docker Space UI.
STARTER_VOICES = [
    "af_heart",
    "af_bella",
    "am_michael",
    "bf_emma",
    "jf_alpha",
    "zf_xiaobei",
    "ef_dora",
    "ff_siwis",
    "dm_martin",
    "df_victoria",
    "diem_trinh",
]

_runtime = None
_runtime_lock = threading.Lock()


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

        # Limit eagerly prepared voices on tiny hosts: the starter set only.
        # The Docker Space keeps the full catalogue; here we trade breadth
        # for cold-start time and RAM on free CPU hardware.
        store = RuntimeSettingsStore()
        try:
            store.set_served_voices(list(STARTER_VOICES))
        except Exception:
            pass
        _runtime = InferenceRuntime(settings=store, eager_voices=True)
        return _runtime


def generate(text: str, voice: str, speed: float):
    text = (text or "").strip()
    if not text:
        raise ValueError("Please enter some text first.")
    speed = max(0.25, min(4.0, float(speed or 1.0)))
    runtime = get_runtime()
    result = runtime.synthesize(
        text=text, voice=voice, speed=speed, device="cpu", input_type="text"
    )
    if result is None:
        raise RuntimeError("Synthesis returned no audio.")
    return SAMPLE_RATE, result.audio


def build_demo():
    import gradio as gr

    with gr.Blocks(title="KokoroTTS-HF (CPU demo)") as demo:
        gr.Markdown(
            "# KokoroTTS-HF community demo\n"
            "Lightweight CPU demo. For the full UI + OpenAI-compatible API, "
            "use the Docker Space flavour (see `docs/HF_SPACES.md`).\n\n"
            "Based on Hangry Labs KokoroTTS (Apache-2.0) and hexgrad Kokoro."
        )
        with gr.Row():
            text = gr.Textbox(
                label="Text", value=DEFAULT_TEXT, lines=5, max_lines=12
            )
        with gr.Row():
            voice = gr.Dropdown(
                label="Voice", choices=STARTER_VOICES, value=DEFAULT_VOICE
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
    demo.queue(max_size=4, concurrency_count=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
