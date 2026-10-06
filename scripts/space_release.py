#!/usr/bin/env python3
"""One-command Space release: stage, upload, wait for RUNNING, verify live.

Usage:
  export HF_TOKEN="hf_..."   # write-access token, never commit it
  python scripts/space_release.py --space-id nam194/kokorotts-hf-cpu

Steps:
  1. Stage the gradio CPU bundle (reuses scripts/deploy_hf_space.py).
  2. Upload to the existing Space (no create_repo: avoids the PRO gate).
  3. Poll get_space_runtime until stage == RUNNING (timeout configurable).
  4. Verify: homepage HTTP 200 + UI markers, /generate returns real audio,
     guest text over 300 chars is rejected before inference.

Exit 0 only when every step passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deploy_hf_space import build_staging  # noqa: E402

POLL_INTERVAL = 20
UI_MARKERS = ("Tạo giọng đọc", "Nghe thử giọng mẫu", "Author key")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release the CPU Space.")
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--timeout-min", type=float, default=15.0)
    parser.add_argument("--verify-text", default="Kiểm tra release tự động.")
    return parser.parse_args(argv)


def wait_running(api, space_id: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        try:
            runtime = api.get_space_runtime(repo_id=space_id)
            stage = runtime.stage
        except Exception as exc:  # transient API errors while rebuilding
            stage = f"API_ERROR:{type(exc).__name__}"
        print(f"[poll {attempt}] stage={stage}", flush=True)
        if stage == "RUNNING":
            print(f"hardware={runtime.hardware}", flush=True)
            return
        if time.time() > deadline:
            raise TimeoutError(f"Space still not RUNNING: last={stage}")
        time.sleep(POLL_INTERVAL)


def homepage_ok(space_id: str, timeout_s: float) -> None:
    user, repo = space_id.split("/")
    url = f"https://{user}-{repo}.hf.space/"
    deadline = time.time() + timeout_s
    # The runtime flips to RUNNING before the new build serves traffic:
    # keep polling the homepage until every UI marker of THIS bundle
    # appears (proves the fresh deploy is the one answering).
    last = None
    while True:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                body = resp.read().decode("utf-8", "replace")
            missing = [m for m in UI_MARKERS if m not in body]
            if resp.status == 200 and not missing:
                print(f"homepage OK: {url}", flush=True)
                return
            last = f"status={resp.status} missing={missing}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        if time.time() > deadline:
            raise RuntimeError(f"homepage check failed: {url} {last}")
        print(f"homepage not fresh yet ({last}), retrying…", flush=True)
        time.sleep(POLL_INTERVAL)


def api_ok(space_id: str, text: str) -> None:
    from gradio_client import Client

    client = Client(space_id)
    out = client.predict(text, "Single", "af_heart", "af_bella",
                         1.0, 0.5, "", api_name="/generate")
    data = open(out, "rb").read()
    assert len(data) > 1000, f"suspiciously small audio: {len(data)}"
    print(f"/generate OK: {len(data)} bytes", flush=True)
    try:
        client.predict("x" * 301, "Single", "af_heart", "af_bella",
                       1.0, 0.5, "", api_name="/generate")
    except Exception as exc:
        if "300" in str(exc):
            print("guest 300-char gate OK", flush=True)
            return
        raise
    raise RuntimeError("guest gate missing: 301-char text was accepted")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.token:
        print("error: set HF_TOKEN env var or pass --token", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfFolder

    api = HfApi(token=args.token or HfFolder.get_token())
    with tempfile.TemporaryDirectory(prefix="kokorotts-hf-space-") as tmp:
        dest = Path(tmp)
        staged = build_staging("gradio", dest)
        print(f"staged {len(staged)} entries", flush=True)
        api.upload_folder(repo_id=args.space_id, repo_type="space",
                          folder_path=str(dest))
        print(f"uploaded https://huggingface.co/spaces/{args.space_id}",
              flush=True)
    wait_running(api, args.space_id, args.timeout_min * 60)
    homepage_ok(args.space_id, args.timeout_min * 60)
    api_ok(args.space_id, args.verify_text)
    print("RELEASE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
