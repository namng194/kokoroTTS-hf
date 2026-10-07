---
title: KokoroTTS-HF Gradio Demo
emoji: 🔊
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.39.0
hf_oauth: true
app_file: app.py
hardware: cpu-basic
pinned: false
license: apache-2.0
short_description: 70-voice CPU TTS demo (Kokoro community edition)
---

# KokoroTTS-HF (Gradio demo)

Full 70-voice demo running on Space CPU. Type text, pick any voice grouped by
language, press **Generate**. Lean-boot and character-guard behaviour follow
`KOKOROTTS_PRELOAD` / `KOKOROTTS_MAX_CHARS`.

Attribution: based on Hangry Labs KokoroTTS (Apache-2.0) and the original
[hexgrad/Kokoro](https://github.com/hexgrad/kokoro) research project.
German voices use kikiri-tts checkpoints; Vietnamese uses the ContextBoxAI
checkpoint — see `THIRD_PARTY_NOTICES.md`.
