"""Full-catalogue Gradio demo for Hugging Face Spaces.

Runs entirely on Space CPU. Serves the complete 70-voice catalogue.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    # Annotation-only import: keeps `gr` lazy at runtime while letting
    # typing.get_type_hints (used by Gradio to detect the OAuth profile
    # parameter) resolve names via module globals (injected in build_demo).
    import gradio as gr

# Access gate: anonymous guests are capped per request; the Space owner
# signs in with Hugging Face (LoginButton + hf_oauth) and is detected
# automatically — no manual key. Owner name is configurable via env.
_GUEST_MAX_CHARS = 300
_OWNER_ENV = "KOKOROTTS_OWNER"
_DEFAULT_OWNER = "nam194"
_LOG_TEXT_CAP = 2000


def space_owner() -> str:
    return os.getenv(_OWNER_ENV, _DEFAULT_OWNER)


def is_owner(profile) -> bool:
    """True only for the signed-in Space owner (None/others → False)."""
    return (
        profile is not None
        and getattr(profile, "username", None) == space_owner()
    )

# Hard defaults: full 70 voices, inference device is always CPU.
os.environ.setdefault("KOKOROTTS_PRELOAD", "all")
os.environ.setdefault("KOKOROTTS_DEVICE", "cpu")
os.environ.setdefault("KOKOROTTS_HF_SPACE", "1")

# Weights live on the Hub model repos (a full bundle exceeds the free Space
# repo cap, see below) and are fetched lazily on first use, then cached.
# KOKOROTTS_BUNDLED_CHECKPOINTS=1 remains available to force local-only
# lookups when checkpoints ARE pre-seeded (e.g. PRO workspaces).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HUB_CACHE"] = os.path.join(_APP_DIR, "hf-cache", "hub")

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


def generate(text: str, mode: str, voice: str, voice_b: str, speed: float, mix: float, oauth_profile: "gr.OAuthProfile | None" = None):
    authorized = is_owner(oauth_profile)
    text = check_text_length(text)
    if not authorized and len(text) > _GUEST_MAX_CHARS:
        text = text[:_GUEST_MAX_CHARS]
    if not text:
        raise ValueError("Please enter some text first.")
    who = getattr(oauth_profile, "username", None) or "guest"
    logged = text if len(text) <= _LOG_TEXT_CAP else (
        text[:_LOG_TEXT_CAP] + f"…[truncated {len(text) - _LOG_TEXT_CAP} chars]"
    )
    logger.info(
        "generate voice={} mode={} speed={} chars={} user={} owner={} text={!r}",
        voice, mode, speed, len(text), who, authorized, logged,
    )
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
  background: linear-gradient(135deg, #0f7665 0%, #0e7490 60%, #0369a1 100%);
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
    logger.info("preview voice={}", voice)
    path = sample_path_for_voice(voice)
    if path is None:
        raise ValueError(f"No bundled sample for voice '{voice}'.")
    return path


def build_demo():
    import gradio as gr

    # Make annotation names resolvable for Gradio's get_type_hints-based
    # special-parameter detection (OAuthProfile injection).
    globals().update({"gr": gr, "OAuthProfile": gr.OAuthProfile})

    summary = profile_summary()
    all_voices = voice_choices()

    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="cyan",
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
        with gr.Row():
            login_btn = gr.LoginButton("Sign in with 🤗 (owner: unlimited)")
            login_status = gr.Markdown(
                "_Browsing as guest — 300 characters per request._"
            )
        owner_state = gr.State(False)

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

        def _count_chars(value: str | None, owner: bool = False):
            capped = value if owner else (value or "")[:_GUEST_MAX_CHARS]
            return capped, f"_{len(capped or '')} characters._"

        def _whoami(oauth_profile: "gr.OAuthProfile | None" = None):
            owner = is_owner(oauth_profile)
            if oauth_profile is not None:
                status = (
                    f"_Signed in as **{oauth_profile.username}** — "
                    + ("unlimited length unlocked._"
                       if owner
                       else "guest limit applies._")
                )
            else:
                status = "_Browsing as guest — 300 characters per request._"
            return status, owner

        demo.load(fn=_whoami, inputs=None, outputs=[login_status, owner_state])
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
        text.change(fn=_count_chars, inputs=[text, owner_state],
                      outputs=[text, counter])
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
