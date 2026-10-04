# Deploying KokoroTTS-HF on Hugging Face Spaces

Two supported flavours. Pick one:

| Flavour | SDK | Best for | Cold start | RAM (typical) |
|---|---|---|---|---|
| **Docker** (recommended) | `docker` | Full product: browser workspace + OpenAI-compatible + native APIs, streaming, 70 voices | ~1–3 min first pull/build | ~2–4 GB |
| **Gradio demo** | `gradio` | One-click free-CPU demo, smallest footprint | ~30–60 s + model download | ~1–2 GB |

Both are CPU-first so they run on the free tier. Kokoro-82M (~82M params) is
lightweight by TTS standards: CPU inference works, GPU just makes it faster.
No VRAM tuning needed on Spaces; locally an 8GB card (e.g. RTX 3070 Ti) is
plenty — see "Local VRAM notes" below.

## Option A — Docker Space (recommended)

Serves the exact same FastAPI + browser UI as local Docker runs.

1. Create a Space:
   - Go to <https://huggingface.co/new-space>
   - Name: e.g. `kokorotts-hf`, SDK: **Docker**, hardware: **CPU Basic** (free).
2. Upload the Space files. Easiest: use the helper script (needs an `HF_TOKEN`
   env var with `write` scope — never commit the token):
   ```bash
   export HF_TOKEN="hf_..."          # write-access token, kept out of git
   python scripts/deploy_hf_space.py --flavour docker --space-id <you>/kokorotts-hf
   ```
   What gets uploaded: `Dockerfile.hf` (as `Dockerfile`), `requirements-hf.txt`,
   `kokorotts/`, `assets/`, `pyproject.toml`, `VERSION`, `LICENSE`,
   `THIRD_PARTY_NOTICES.md`, and the Docker Space `README.md` frontmatter.
3. The Space builds automatically and serves port `7860`:
   - UI: `https://<you>-kokorotts-hf.hf.space/`
   - Health: `/health/ready` · Docs: `/tts/docs`
   - OpenAI-compatible: `POST /v1/audio/speech`
   ```bash
   curl -X POST "https://<you>-kokorotts-hf.hf.space/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -d '{"model":"kokoro","input":"Hello from Spaces.","voice":"af_heart"}' \
     -o output.mp3
   ```

Notes:
- First generation downloads weights (~300MB for Kokoro-82M + voice packs)
  into `/data/hf-cache` (the only persistent dir on Spaces). Later requests
  reuse them.
- Optional env vars on the Space: `KOKOROTTS_API_KEY` (Bearer auth for
  `/v1/*`), `HF_TOKEN` (only if a gated asset ever needs it — default assets
  are public, leave unset).
- Upgrades: re-run the deploy script; the Space rebuilds.
- ZeroGPU/GPU upgrade (paid): the same image runs, just faster. If you switch
  a Space to GPU hardware, set `KOKOROTTS_DEVICE=auto` in the Space secrets to
  let it use CUDA; on CPU hardware keep `cpu` (default in `Dockerfile.hf`).

## Option B — Gradio demo Space (free CPU, minimal)

1. Create a Space with SDK **Gradio**, hardware **CPU Basic**.
2. Upload `spaces/gradio_app.py`, `spaces/requirements-gradio.txt` (as
   `requirements.txt`), `requirements-hf.txt`, `kokorotts/`, `VERSION`,
   `LICENSE`, and `spaces/README.md` (as `README.md`):
   ```bash
   export HF_TOKEN="hf_..."
   python scripts/deploy_hf_space.py --flavour gradio --space-id <you>/kokorotts-hf-demo
   ```
3. Open the Space, type text, pick a starter voice, press **Generate**.

Limits of the demo flavour: starter voice set only, WAV output, no streaming,
no OpenAI API. It is a teaser for the Docker flavour.

## Local VRAM notes (RTX 3070 Ti 8GB and similar)

- Docker GPU run (full local product):
  ```bash
  docker run -p 7860:7860 --gpus all kokorotts-hf:local
  ```
- Official VRAM baseline of the upstream line (quiet RTX 5070 Ti, all 70
  voices, sequential single requests): ~1.7GB allocated / ~2.1GB reserved peak
  for the Kokoro process. An 8GB card comfortably fits any single model family.
- Keep it light: use one voice family at a time via System tab →
  model-pack settings (standard / German Martin / German Victoria /
  Vietnamese), or set `KOKOROTTS_DEVICE=cpu` to skip the GPU entirely for
  smoke tests.
- The deploy helper and CI checks in this repo never touch the GPU.

## Troubleshooting

- **Build OOM on free tier**: use the Gradio flavour, or keep the Docker
  flavour but avoid baking weights into the image (this repo's `Dockerfile.hf`
  already downloads lazily — do not `COPY` model files into it).
- **First request slow**: expected — weights download once, then cache.
- **401 from api/whoami with a fresh token**: create a new fine-grained token
  with `Make calls to the serverless Inference API` off but **Spaces write**
  on, and retry. Never paste tokens into Issues.
- **German/Vietnamese voices fail on a fresh Space**: they lazy-load extra
  checkpoints (kikiri-tts / ContextBoxAI). Generate once per family to warm
  the cache; check `/tts/status` for loaded models.
