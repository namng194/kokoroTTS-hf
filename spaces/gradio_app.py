"""Full-catalogue Gradio demo for Hugging Face Spaces.

Runs entirely on Space CPU. Serves the complete 70-voice catalogue.
"""

from __future__ import annotations

import os
import threading

# Hard defaults: full 70 voices, inference device is always CPU.
os.environ.setdefault("KOKOROTTS_PRELOAD", "all")
os.environ.setdefault("KOKOROTTS_DEVICE", "cpu")
os.environ.setdefault("KOKOROTTS_HF_SPACE", "1")

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
        try:
            _runtime.set_served_model_families(ALL_FAMILIES)
        except Exception:
            pass
        return _runtime


def _inference_device() -> str:
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


_LANG_ALL = "All languages"

# Pre-generated voice samples bundled next to app.py (samples/ dir):
# previews play these files directly — zero inference compute.
_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def sample_path_for_voice(voice: str) -> str | None:
    """Filesystem path of the bundled sample mp3 for a voice, if present."""
    candidate = os.path.join(_SAMPLES_DIR, f"kokorotts-{voice}.mp3")
    return candidate if os.path.isfile(candidate) else None

_FONT_HEAD = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link href='https://fonts.googleapis.com/css2?"
    "family=Be+Vietnam+Pro:wght@400;500;600;700&family=Inter:wght@400;500;600;700"
    "&display=swap' rel='stylesheet'>"
)

_BANNER_CSS = """
.kokoro-banner {
  background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 55%, #2563eb 100%);
  border-radius: 16px;
  padding: 28px 32px;
  color: #fff;
  margin-bottom: 16px;
}
.kokoro-banner h1 { margin: 0 0 6px 0; font-size: 1.9rem; }
.kokoro-banner p { margin: 0; opacity: 0.92; }
.kokoro-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.kokoro-badge {
  background: rgba(255, 255, 255, 0.16);
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: 999px;
  padding: 4px 14px;
  font-size: 0.82rem;
  font-weight: 600;
}
.kokoro-footer { text-align: center; opacity: 0.75; font-size: 0.85rem; }
.gradio-container, .gradio-container * {
  font-family: 'Be Vietnam Pro', 'Inter', system-ui, -apple-system,
    'Segoe UI', sans-serif !important;
}
#sample-voice-list { max-height: 300px; overflow-y: auto; }
"""


def language_options() -> list[str]:
    """Dropdown options: every language plus an unfiltered entry."""
    return [_LANG_ALL] + [
        f"{LANGUAGE_CHOICES[code]} ({code})" for code in LANGUAGE_CHOICES
    ]


def voices_for_language(option: str) -> list[tuple[str, str]]:
    """Voice (label, id) pairs filtered by a language dropdown option."""
    if option == _LANG_ALL:
        return voice_choices()
    code = option.rsplit("(", 1)[-1].rstrip(")")
    return [
        (label, voice_id)
        for label, voice_id in voice_choices()
        if voice_language(voice_id) == code
    ]


def preview_voice(voice: str, speed: float):
    """Play the bundled sample file for a voice — no inference, instant."""
    path = sample_path_for_voice(voice)
    if path is None:
        raise ValueError(f"No bundled sample for voice '{voice}'.")
    return path


def build_demo():
    import gradio as gr

    summary = profile_summary()
    limit = summary["max_chars"]
    guard_line = (
        f"Limit: {limit} characters per request."
        if limit else "No per-request character limit on this deployment."
    )
    all_voices = voice_choices()

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        font="Be Vietnam Pro",
    )
    with gr.Blocks(
        title="KokoroTTS-HF — 70 voices on CPU",
        theme=theme,
        css=_BANNER_CSS,
        head=_FONT_HEAD,
    ) as demo:
        gr.HTML(
            "<div class='kokoro-banner'>"
            "<h1>🔊 KokoroTTS-HF</h1>"
            "<p>Full 70-voice text-to-speech catalogue running on Space CPU. "
            "Generate your audio above, audition every sample file below.</p>"
            "<div class='kokoro-badges'>"
            "<span class='kokoro-badge'>70 voices</span>"
            "<span class='kokoro-badge'>11 languages</span>"
            "<span class='kokoro-badge'>CPU inference</span>"
            "<span class='kokoro-badge'>24 kHz output</span>"
            "</div></div>"
        )

        gr.Markdown("## 🔊 Tạo giọng đọc")
        gr.Markdown(
            "_Synthesise any text on Space CPU. Type text, pick a voice, "
            "press Generate._"
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Text to speak",
                    value=DEFAULT_TEXT,
                    lines=5,
                    max_lines=12,
                    placeholder="Type or paste any text here…",
                )
                counter = gr.Markdown(
                    f"_{len(DEFAULT_TEXT)} characters._"
                )
                with gr.Row():
                    mode = gr.Radio(
                        label="Mode",
                        choices=["Single", "Blend"],
                        value="Single",
                    )
                    lang = gr.Dropdown(
                        label="Language filter",
                        choices=language_options(),
                        value=_LANG_ALL,
                    )
                voice = gr.Dropdown(
                    label="Voice A",
                    choices=all_voices,
                    value=DEFAULT_VOICE,
                    filterable=True,
                )
                with gr.Row(visible=False) as blend_row:
                    voice_b = gr.Dropdown(
                        label="Voice B (must share A's model family)",
                        choices=all_voices,
                        value="af_bella",
                        filterable=True,
                    )
                    mix = gr.Slider(
                        label="Mix: weight of voice B "
                              "(0 = pure A, 1 = pure B)",
                        minimum=0.0, maximum=1.0,
                        step=0.05, value=0.5,
                    )
                with gr.Row():
                    speed = gr.Slider(
                        label="Speed", minimum=0.25, maximum=4.0,
                        step=0.05, value=1.0,
                    )
                with gr.Row():
                    btn = gr.Button(
                        "🔊 Generate", variant="primary", scale=2
                    )
                    clear = gr.ClearButton([text], value="Clear text")

            with gr.Column(scale=2):
                audio = gr.Audio(
                    label="Output (24 kHz WAV)", interactive=False
                )
                gr.Markdown(
                    f"_{guard_line} Profile: `{summary['preload']}`._"
                )
                with gr.Accordion(
                    "How voice blending works", open=False
                ):
                    gr.Markdown(
                        "_Blend_ mixes two voices of the **same model "
                        "family** into a new synthetic speaker — a "
                        "KokoroTTS-HF exclusive. Pick **Blend** mode, "
                        "choose voice B, and set how much of B to "
                        "mix in."
                    )
                with gr.Accordion(
                    "Tips for best quality", open=False
                ):
                    gr.Markdown(
                        "- Short sentences synthesise fastest.\n"
                        "- Keep speed between 0.9 and 1.1 for "
                        "natural pacing.\n"
                        "- Vietnamese, German, Japanese and Chinese "
                        "voices load their model pack on first use — "
                        "the first request takes longer, later ones "
                        "are fast."
                    )

        gr.Examples(
            examples=[
                ["Hello from KokoroTTS-HF on Space CPU.", "Single", "af_heart", "af_bella", 1.0, 0.5],
                ["Xin chào, đây là giọng đọc tiếng Việt chạy hoàn toàn trên CPU.", "Single", "diem_trinh", "af_bella", 1.0, 0.5],
                ["Two voices become one.", "Blend", "af_heart", "af_bella", 1.0, 0.5],
            ],
            inputs=[text, mode, voice, voice_b, speed, mix],
            label="Try an example",
        )

        gr.Markdown("---")
        gr.Markdown("## 🎧 Nghe thử giọng mẫu")
        gr.Markdown(
            "_Pick a voice from the list — its bundled sample plays "
            "below instantly, no compute used. Then use it in the "
            "generator above._"
        )
        lang_sample = gr.Dropdown(
            label="Language filter",
            choices=language_options(),
            value=_LANG_ALL,
        )
        with gr.Row():
            with gr.Column(scale=2):
                sample_voice = gr.Radio(
                    label="Voices (click to audition)",
                    choices=all_voices,
                    value=DEFAULT_VOICE,
                    elem_id="sample-voice-list",
                )
            with gr.Column(scale=3):
                now_playing = gr.Markdown(
                    f"_Now auditioning: **{DEFAULT_VOICE}**_"
                )
                preview = gr.Audio(
                    label="🎧 Voice sample (bundled file)",
                    interactive=False,
                    autoplay=True,
                )
        gr.Markdown(
            "<div class='kokoro-footer'>Based on Hangry Labs KokoroTTS "
            "(Apache-2.0) and hexgrad Kokoro. German voices use kikiri-tts "
            "checkpoints; Vietnamese uses the ContextBoxAI checkpoint.</div>"
        )

        def _toggle_blend(selected: str):
            return gr.Row(visible=(selected == "Blend"))

        def _filter_sample_voices(option: str, current: str):
            options = voices_for_language(option)
            ids = {voice_id for _, voice_id in options}
            keep = current if current in ids else (
                options[0][1] if options else None
            )
            return gr.Radio(choices=options, value=keep)

        def _audition(selected: str):
            return (
                preview_voice(selected, 1.0),
                f"_Now auditioning: **{selected}**_",
            )

        def _filter_voices(option: str, current_a: str, current_b: str):
            options = voices_for_language(option)
            ids = {voice_id for _, voice_id in options}
            keep_a = current_a if current_a in ids else (
                options[0][1] if options else None
            )
            keep_b = current_b if current_b in ids else (
                options[0][1] if options else None
            )
            return (
                gr.Dropdown(choices=options, value=keep_a),
                gr.Dropdown(choices=options, value=keep_b),
            )

        def _count_chars(value: str | None):
            return f"_{len(value or '')} characters._"

        mode.change(fn=_toggle_blend, inputs=mode, outputs=blend_row)
        lang_sample.change(
            fn=_filter_sample_voices, inputs=[lang_sample, sample_voice],
            outputs=sample_voice,
        )
        sample_voice.change(
            fn=_audition, inputs=sample_voice,
            outputs=[preview, now_playing],
        )
        lang.change(
            fn=_filter_voices, inputs=[lang, voice, voice_b],
            outputs=[voice, voice_b],
        )
        text.change(fn=_count_chars, inputs=text, outputs=counter)
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
