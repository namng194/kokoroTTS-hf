# Deploying KokoroTTS-HF on Hugging Face Spaces

Live now (free tier, static gallery, no backend):
**https://nam194-kokorotts-hf.static.hf.space/** — all 70 voices playable,
deployed from this repo with `scripts/deploy_hf_space.py --flavour static`.

Live now (free, real inference, no server):
**https://nam194-kokorotts-hf.static.hf.space/#studio** — KokoroTTS-HF
Studio runs Kokoro-82M 100% in the visitor's browser (kokoro-js q8,
WebGPU/WASM, verified end-to-end in Node against the same model). No quota,
no uploads, offline after the ~90MB first download. English voices fully
supported in-browser.

> Server-GPU Spaces were retired from this project's free-tier strategy:
> free ZeroGPU quota (≈5 min/day) cannot sustain a public demo, and hosted
> CPU Gradio/Docker Spaces need PRO. The Gradio (`--flavour gradio`) and
> Docker (`Dockerfile.hf`) flavours below remain ready for PRO workspaces
> and self-hosting (CPU inference verified locally: 0 MB VRAM).

### Voice blending (KokoroTTS-HF exclusive)

Mix two same-family voices into a new synthetic speaker. `blend` is the
weight of `voice_b` (`0.0` = pure `voice`, `1.0` = pure `voice_b`):

```bash
curl -X POST "http://localhost:7860/tts/blend" \
  -H "Content-Type: application/json" \
  -d '{"text":"Two voices become one.","voice":"af_heart","voice_b":"af_bella","blend":0.5}' \
  -o blend.wav
```

Cross-family pairs return HTTP 400 (different weight spaces cannot mix).

> Pricing reality (Oct 2026): Gradio/Docker Spaces on hosted CPU now require
> a PRO subscription (creation fails with 402 otherwise). **ZeroGPU
> (`zero-a10g`) still works on free accounts** and is the recommended hosted
> inference target: it mandates `@spaces.GPU` (hence `--torch cuda`) and
> Python 3.10 on the builder (hence the held-back pins in
> `requirements-hf.txt` plus `spaces/packages.txt` for the source-only
> pyopenjtalk build).

Three supported flavours. Pick one:

| Flavour | SDK | Best for | Cold start | RAM (typical) |
|---|---|---|---|---|
| **Static gallery** (live, free) | `static` | Public voice previews, zero backend cost | instant | ~0 |
| **Docker** (needs PRO hosted) | `docker` | Full product: browser workspace + OpenAI-compatible + native APIs, streaming, 70 voices | ~1–3 min first pull/build | ~2–4 GB |
| **Gradio demo** (needs PRO hosted) | `gradio` | One-click CPU demo, smallest dynamic footprint | ~30–60 s + model download | ~1–2 GB |

## Option 0 — Static gallery Space (free, live)

No backend, no PRO needed. Serves the `examples/` voice gallery page:

```bash
export HF_TOKEN="hf_..."   # any write token; static hosting is free
python scripts/deploy_hf_space.py --flavour static --space-id YOU/kokoroTTS-hf
```

This stages `examples/index.html` at the Space root plus its JS, all voice
MP3s, and the two small referenced assets — then creates/uploads the Space.
Note the serving subdomain for static Spaces ends in
`.static.hf.space`, e.g. `https://YOU-kokorotts-hf.static.hf.space/`.

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
- **Lean boot**: the Space image sets `KOKOROTTS_PRELOAD=standard`, so it
  boots with the 54 standard voices eager and loads the German/Vietnamese
  checkpoints only when you enable those families (System tab → model packs,
  or `PUT /system/settings/model-families`). Choices persist in `/data`.
  Set `KOKOROTTS_PRELOAD=all` for the full 70-voice eager boot, or `lazy`
  for the smallest possible cold start (every voice loads on first use).
- **Abuse guard**: `KOKOROTTS_MAX_CHARS=5000` rejects oversized requests
  with HTTP 400 (OpenAI-shaped error on `/v1/*`). Unset it for unlimited,
  exactly like local Docker runs.
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

Limits of the demo flavour: WAV output only, no streaming, no OpenAI API,
one request at a time. It is a teaser for the Docker flavour.

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

## Verified: pure-CPU inference (0 MB VRAM)

Measured Oct 2026 on a 20-core laptop CPU, first run (weights downloaded
once, ~330MB for the standard family):

| Request | Audio | Wall clock | Device |
|---|---|---|---|
| `af_heart` "Hello from CPU inference on a laptop." | 3.17 s | 15.3 s cold / <1 s warm | cpu |
| Blend `af_heart`+`af_bella` @0.5 (warm) | 2.38 s | 0.7 s | cpu |
| `diem_trinh` Vietnamese (warm model, cold family) | 3.62 s | 16.9 s | cpu |

Cold family/model loads dominate; warm synthesis runs ~0.2–3x realtime on
CPU. Conclusion for constrained hosts: CPU inference is slow but fully
working — a valid fallback when no GPU and no ZeroGPU quota remain.
The `spaces/gradio_app.py` auto-selects CUDA when present, CPU otherwise.

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
