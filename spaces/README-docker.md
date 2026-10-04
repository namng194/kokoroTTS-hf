---
title: KokoroTTS-HF
emoji: 🔊
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: KokoroTTS UI + OpenAI-compatible API (CPU Space)
---

# KokoroTTS-HF (Docker flavour)

Full community edition: responsive browser workspace + OpenAI-compatible
`POST /v1/audio/speech` + native `POST /tts/generate` HTTP APIs on port `7860`.

This Space runs `Dockerfile.hf` (CPU-only torch, lazy model download, cache in
`/data`). First generation downloads the Kokoro-82M weights (~300MB); later
requests reuse the cache.

Deploy your own copy: see `docs/HF_SPACES.md` or run
`scripts/deploy_hf_space.py --flavour docker`.

Attribution: based on Hangry Labs KokoroTTS (Apache-2.0) and
[hexgrad/Kokoro](https://github.com/hexgrad/kokoro). See
`THIRD_PARTY_NOTICES.md` for model/voice asset credits.
