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

> Strategy: every Space flavour in this repo runs on Space CPU.
> The Gradio demo (`--flavour gradio`) serves all 70 voices with
> `device=cpu` and `KOKOROTTS_PRELOAD=all` (verified: 0 MB VRAM).

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

Two supported flavours. Pick one:

| Flavour | SDK | Best for | Cold start | RAM (typical) |
|---|---|---|---|---|
| **Static gallery** | `static` | Public voice previews, zero backend cost | instant | ~0 |
| **Gradio demo** | `gradio` | Full 70-voice CPU synthesis in one click | ~30–60 s + model download | ~1–2 GB |

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

Both flavours are CPU-first. Kokoro-82M (~82M params) is
lightweight by TTS standards: CPU inference works end to end.
No VRAM tuning needed on Spaces; locally an 8GB card (e.g. RTX 3070 Ti) is
plenty — see "Local VRAM notes" below.

## Option A — Gradio demo Space (CPU, full 70 voices)

1. Create a Space with SDK **Gradio**, hardware **CPU**.
2. Upload the bundle with the helper (needs an `HF_TOKEN`
   env var with `write` scope — never commit the token):
   ```bash
   export HF_TOKEN="hf_..."          # write-access token, kept out of git
   python scripts/deploy_hf_space.py --flavour gradio --space-id <you>/kokorotts-hf
   ```
   What gets uploaded: `spaces/gradio_app.py` (as `app.py`),
   `spaces/requirements-gradio.txt` (flattened as `requirements.txt`),
   `requirements-hf.txt`, `spaces/packages.txt` (espeak-ng/ffmpeg toolchains),
   `kokorotts/`, `VERSION`, `LICENSE`, and the Space `README.md` frontmatter.
3. The Space builds automatically and serves port `7860`:
   - UI: `https://<you>-kokorotts-hf.hf.space/`
4. Open the Space, type text, pick any of the 70 voices grouped by
   language, press **Generate**. Blend mode mixes two same-family voices.

Notes:
- First generation downloads weights (~300MB for Kokoro-82M + voice packs);
  later requests reuse the cache.
- **Full catalogue boot**: the Space image sets `KOKOROTTS_PRELOAD=all`, so
  all 70 voices (standard + German + Vietnamese families) are served.
  `KOKOROTTS_MAX_CHARS` guards oversized requests with HTTP 400.
- Upgrades: re-run the deploy script; the Space rebuilds.

Limits of the demo flavour: WAV output only, no streaming, no OpenAI API,
one request at a time.

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
working. `spaces/gradio_app.py` always runs inference with `device="cpu"`.

## Troubleshooting

- **Build OOM**: keep the gradio bundle lean (this repo's Space requirements
  already download lazily — do not bake model files into it).
- **First request slow**: expected — weights download once, then cache.
- **401 from api/whoami with a fresh token**: create a new fine-grained token
  with `Make calls to the serverless Inference API` off but **Spaces write**
  on, and retry. Never paste tokens into Issues.
- **German/Vietnamese voices fail on a fresh Space**: they lazy-load extra
  checkpoints (kikiri-tts / ContextBoxAI). Generate once per family to warm
  the cache.
