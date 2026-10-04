---
title: KokoroTTS-HF Gradio Demo
emoji: 🔊
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.39.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 70-voice CPU TTS demo (Kokoro community edition)
---

# KokoroTTS-HF (Gradio demo flavour)

Full 70-voice demo on free CPU. Type text, pick any voice grouped by
language, press **Generate**. Lean-boot and character-guard behaviour follow
`KOKOROTTS_PRELOAD` / `KOKOROTTS_MAX_CHARS` (see `docs/HF_SPACES.md`).

For the full product (browser workspace, OpenAI-compatible + native HTTP APIs,
70 voices, streaming, GPU support), deploy the **Docker flavour** instead:

- Image recipe: `Dockerfile.hf` (CPU-optimized, serves port `7860`)
- Guide: `docs/HF_SPACES.md`
- Deploy helper: `scripts/deploy_hf_space.py --flavour docker`

Attribution: based on Hangry Labs KokoroTTS (Apache-2.0) and the original
[hexgrad/Kokoro](https://github.com/hexgrad/kokoro) research project.
German voices use kikiri-tts checkpoints; Vietnamese uses the ContextBoxAI
checkpoint — see `THIRD_PARTY_NOTICES.md`.
