# Vision: the deploy-anywhere lightweight edition

Upstream (Hangry Labs KokoroTTS) optimizes the **full local Docker product**:
everything baked/eager, GPU first. This fork optimizes a different target and
does not try to out-do upstream at its own game:

> **Run the same 70 voices on whatever you have** — a free CPU Space, an 8 GB
> VRAM laptop, a CPU-only VPS — with graceful degradation instead of failure.

## Shipped (this fork's own contributions)

1. **Hugging Face Spaces, Space-only** — a live free-tier static gallery
   plus **KokoroTTS-HF Studio**: real Kokoro-82M inference 100% in the
   visitor's browser (kokoro-js, WebGPU/WASM, no server, no quota).
   Server-GPU Spaces were retired: free ZeroGPU quota (≈5 min/day) cannot
   sustain a public demo. Guide: `docs/HF_SPACES.md`.
   Deploy: `scripts/deploy_hf_space.py`.
2. **Constrained-host profiles** (`kokorotts/space.py`, wired into `api.py`
   and `openai_compat.py`):
   - `KOKOROTTS_PRELOAD=all|standard|lazy` — lean the boot; default `all`
     preserves upstream behavior byte-for-byte, so this is strictly additive.
   - `KOKOROTTS_MAX_CHARS` — per-request text guard (HTTP 400 native,
     OpenAI-shaped error on `/v1/*`) for shared/public deployments.
   - Operators can still enable every family at runtime via the System tab;
     choices persist in `/data` on Spaces.
3. **CI + secret hygiene** — `hf-space-check` workflow validates both Space
   flavours without GPU or downloads and rejects committed tokens.
4. **Voice blending** (`kokorotts/voice_blend.py`, `POST /tts/blend`, Blend
   mode on the live Space) — mix any two same-family voices into a new
   synthetic speaker. Upstream serves fixed voices only.

## Roadmap (planned, not yet built)

- CPU latency: int8/quantized inference path behind an env flag, measured
  with the existing `benchmarks/tts` harness before/after.
- Concurrency guard: bounded in-flight synthesis queue with HTTP 429 +
  `Retry-After` for free-tier Spaces (today: fail only on OOM).
- Streaming-first demo: chunked Gradio audio output instead of one-shot.
- ARM64 image for Apple Silicon / Raspberry Pi-class hosts.
- Space auto-detect: read `SPACE_HARDWARE` to pick `standard` vs `lazy`
  without manual env configuration.

## Non-goals

- Forking the model or voice quality work — that belongs upstream
  (`hexgrad/Kokoro`, kikiri-tts, ContextBoxAI).
- Baking weights into Space images — lazy download + `/data` cache keeps
  builds small and free-tier compatible.
- Tracking upstream file-for-file — divergence is intentional and
  documented per change above.
