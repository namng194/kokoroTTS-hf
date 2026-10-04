---
title: KokoroTTS-HF Voice Gallery
emoji: 🔊
colorFrom: purple
colorTo: indigo
sdk: static
pinned: false
license: apache-2.0
short_description: 70 KokoroTTS voices gallery + self-host links
---

# KokoroTTS-HF Voice Gallery (free-tier Space)

Static gallery — no backend needed, free forever. Listen to every voice,
then run the real thing yourself:

- **Repo**: <https://github.com/namng194/kokoroTTS-hf>
- **Local Docker (GPU)**: `docker run -p 7860:7860 --gpus all $(see README for image tags)`
- **Full guide**: `docs/HF_SPACES.md` in the repo (Docker + Gradio flavours,
  lean-boot profiles, self-hosting)

> Live synthesis (Docker/Gradio Space flavours) needs a PRO subscription on
> current Hugging Face pricing; this static gallery stays up for everyone.
> With PRO, deploy the same repo via `scripts/deploy_hf_space.py
> --flavour docker|gradio`.

Attribution: based on Hangry Labs KokoroTTS (Apache-2.0) and
[hexgrad/Kokoro](https://github.com/hexgrad/kokoro). Voice/model credits in
`THIRD_PARTY_NOTICES.md`.
