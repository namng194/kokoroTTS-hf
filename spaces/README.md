---
title: KokoroTTS-HF Gradio Demo
emoji: 🔊
colorFrom: violet
colorTo: indigo
sdk: gradio
sdk_version: 5.39.0
app_file: gradio_app.py
pinned: false
license: apache-2.0
short_description: Lightweight CPU text-to-speech demo (Kokoro-82M community edition)
---

# KokoroTTS-HF (Gradio demo flavour)

Lightweight free-CPU demo. Type text, pick a starter voice, press **Generate**.

For the full product (browser workspace, OpenAI-compatible + native HTTP APIs,
70 voices, streaming, GPU support), deploy the **Docker flavour** instead:

- Image recipe: `Dockerfile.hf` (CPU-optimized, serves port `7860`)
- Guide: `docs/HF_SPACES.md`
- Deploy helper: `scripts/deploy_hf_space.py --flavour docker`

Attribution: based on Hangry Labs KokoroTTS (Apache-2.0) and the original
[hexgrad/Kokoro](https://github.com/hexgrad/kokoro) research project.
German voices use kikiri-tts checkpoints; Vietnamese uses the ContextBoxAI
checkpoint — see `THIRD_PARTY_NOTICES.md`.
