#!/usr/bin/env python3
"""Create or update a Hugging Face Space for KokoroTTS-HF.

Flavours:
  docker  - full product (browser UI + OpenAI/native APIs), CPU-optimized.
            Uploads Dockerfile.hf as Dockerfile + runtime files.
  gradio  - lightweight free-CPU demo (spaces/gradio_app.py).
  static  - free-tier voice gallery (examples/ page + mp3s, no backend).
            The only flavour that works without a PRO subscription.

Auth: read from HF_TOKEN env var (or --token, never commit it).
The bundled HF token in the original request is treated as compromised the
moment it was pasted into chat: rotate it at
https://huggingface.co/settings/tokens and use the fresh value here.

Examples:
  export HF_TOKEN="hf_..."
  python scripts/deploy_hf_space.py --flavour docker --space-id YOU/kokorotts-hf
  python scripts/deploy_hf_space.py --flavour docker --space-id YOU/kokorotts-hf --dry-run
  python scripts/deploy_hf_space.py --flavour gradio --space-id YOU/kokorotts-hf-demo
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCKER_FILES = [
    ("Dockerfile.hf", "Dockerfile"),
    ("requirements-hf.txt", "requirements-hf.txt"),
    ("pyproject.toml", "pyproject.toml"),
    ("VERSION", "VERSION"),
    ("LICENSE", "LICENSE"),
    ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ("spaces/README-docker.md", "README.md"),
]
DOCKER_DIRS = ["kokorotts", "assets", "scripts"]

GRADIO_FILES = [
    ("spaces/gradio_app.py", "app.py"),
    ("spaces/requirements-gradio.txt", "requirements.txt"),
    ("spaces/README.md", "README.md"),
    ("requirements-hf.txt", "requirements-hf.txt"),
    ("VERSION", "VERSION"),
    ("LICENSE", "LICENSE"),
]
GRADIO_DIRS = ["kokorotts"]

# Static gallery: examples/ page flattened to the Space root (index.html
# must sit at root) plus the two small assets it references relatively.
STATIC_README = ("spaces/README-static.md", "README.md")
STATIC_ASSETS = ["favicon_small.png", "logo_small.png"]


def build_staging_static(dest: Path) -> list[str]:
    staged: list[str] = []
    src, name = STATIC_README
    shutil.copy2(REPO_ROOT / src, dest / name)
    staged.append(name)
    examples = REPO_ROOT / "examples"
    for item in sorted(examples.iterdir()):
        if item.is_file():
            shutil.copy2(item, dest / item.name)
            staged.append(item.name)
    assets_dir = dest / "assets"
    assets_dir.mkdir(exist_ok=True)
    for asset in STATIC_ASSETS:
        shutil.copy2(REPO_ROOT / "assets" / asset, assets_dir / asset)
        staged.append(f"assets/{asset}")
    return staged


def build_staging(flavour: str, dest: Path) -> list[str]:
    files = DOCKER_FILES if flavour == "docker" else GRADIO_FILES
    dirs = DOCKER_DIRS if flavour == "docker" else GRADIO_DIRS
    staged: list[str] = []
    for src, name in files:
        src_path = REPO_ROOT / src
        if not src_path.exists():
            # install_open_jtalk_dictionary.py lives under scripts/ and is
            # referenced by Dockerfile.hf; keep the tree importable instead
            # of failing when optional files are absent.
            if src_path.name == ".gitkeep":
                continue
            raise FileNotFoundError(f"Required file missing: {src}")
        shutil.copy2(src_path, dest / name)
        staged.append(name)
    for dirname in dirs:
        src_dir = REPO_ROOT / dirname
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Required directory missing: {dirname}/")
        shutil.copytree(src_dir, dest / dirname,
                        ignore=shutil.ignore_patterns("__pycache__"))
        staged.append(dirname + "/")
    return staged


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy KokoroTTS-HF to HF Spaces.")
    parser.add_argument("--flavour", choices=["docker", "gradio", "static"], default="docker")
    parser.add_argument("--space-id", required=True, help="e.g. YOU/kokorotts-hf")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="defaults to $HF_TOKEN")
    parser.add_argument("--dry-run", action="store_true", help="stage files only, no upload")
    parser.add_argument("--stage-dir", default="",
                        help="copy staged Space files into this directory and keep them "
                             "(for manual web drag-drop upload); implies no upload")
    parser.add_argument("--private", action="store_true", help="create a private Space")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.token and not args.dry_run and not args.stage_dir:
        print("error: set HF_TOKEN env var or pass --token (never commit it)", file=sys.stderr)
        return 2
    if args.stage_dir:
        dest = Path(args.stage_dir)
        dest.mkdir(parents=True, exist_ok=True)
        staged = (
            build_staging_static(dest)
            if args.flavour == "static"
            else build_staging(args.flavour, dest)
        )
        print(f"flavour: {args.flavour}")
        print(f"staged {len(staged)} entries into {dest}:")
        for entry in staged:
            print(f"  - {entry}")
        print("Upload this folder's contents to an empty Space via the web UI.")
        return 0
    with tempfile.TemporaryDirectory(prefix="kokorotts-hf-space-") as tmp:
        dest = Path(tmp)
        staged = (
            build_staging_static(dest)
            if args.flavour == "static"
            else build_staging(args.flavour, dest)
        )
        print(f"flavour: {args.flavour}")
        print(f"space:   {args.space_id}")
        print(f"staged {len(staged)} entries:")
        for entry in staged:
            print(f"  - {entry}")
        if args.dry_run:
            print("dry-run: staged only, nothing uploaded.")
            return 0
        from huggingface_hub import HfApi

        api = HfApi(token=args.token)
        api.create_repo(
            repo_id=args.space_id, repo_type="space", space_sdk=args.flavour,
            private=args.private, exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.space_id, repo_type="space", folder_path=str(dest),
        )
        print(f"https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
