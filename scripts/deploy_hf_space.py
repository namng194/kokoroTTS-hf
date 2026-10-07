#!/usr/bin/env python3
"""Create or update a Hugging Face Space for KokoroTTS-HF.

Flavours (all run on Space CPU):
  gradio  - full 70-voice demo (spaces/gradio_app.py).
  static  - voice gallery (examples/ page, no backend).

Auth: read from HF_TOKEN env var (or --token, never commit it).
If a token was ever pasted into chat, treat it as compromised: rotate it at
https://huggingface.co/settings/tokens and use the fresh value here.

Examples:
  export HF_TOKEN="hf_..."
  python scripts/deploy_hf_space.py --flavour gradio --space-id YOU/kokorotts-hf
  python scripts/deploy_hf_space.py --flavour gradio --space-id YOU/kokorotts-hf --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Checkpoint families baked into the Space repo: the app runs with
# HF_HUB_OFFLINE=1 and reads only from hf-cache/hub/ staged below.
# Mirrors the families in kokorotts/catalog.py MODEL_FAMILY_CHOICES.
MODEL_CACHE_DIRS = [
    "models--hexgrad--Kokoro-82M",
    "models--kikiri-tts--kikiri-german-martin",
    "models--kikiri-tts--kikiri-german-victoria",
    "models--contextboxai--Kokoro-Vietnamese",
]
MODELS_DEST = "hf-cache/hub"


def local_hub_cache() -> Path:
    override = os.getenv("HF_HUB_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "huggingface" / "hub"


def stage_models(dest: Path) -> list[str]:
    """Copy checkpoint cache dirs into dest/hf-cache/hub/ (Space repo)."""
    src_root = local_hub_cache()
    hub_dest = dest / MODELS_DEST
    hub_dest.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for dirname in MODEL_CACHE_DIRS:
        src = src_root / dirname
        if not src.is_dir():
            raise FileNotFoundError(
                f"Checkpoint cache missing: {src} "
                "(run one local inference first to populate it)"
            )
        shutil.copytree(src, hub_dest / dirname,
                        ignore=shutil.ignore_patterns(".locks"))
        staged.append(f"{MODELS_DEST}/{dirname}/")
    return staged

GRADIO_FILES = [
    ("spaces/gradio_app.py", "app.py"),
    ("spaces/requirements-gradio.txt", "requirements.txt"),
    ("spaces/README.md", "README.md"),
    ("spaces/packages.txt", "packages.txt"),
    ("requirements-hf.txt", "requirements-hf.txt"),
    ("VERSION", "VERSION"),
    ("LICENSE", "LICENSE"),
]
GRADIO_DIRS = ["kokorotts"]

# Pre-generated voice samples (examples/kokorotts-<voice>.mp3): the Space
# plays these files directly for previews — zero inference compute.
GRADIO_SAMPLE_GLOB = "examples/kokorotts-*.mp3"
GRADIO_SAMPLE_DEST = "samples"

# spaces/requirements-gradio.txt references ../requirements-hf.txt, which
# breaks once staged flat at the Space root. Flatten it at stage time.
GRADIO_PIN_FILE = "spaces/requirements-gradio.txt"
HF_REQUIREMENTS_FILE = "requirements-hf.txt"


def split_requirement_lines(path: str) -> tuple[list[str], list[str]]:
    """Returns (index_lines, package_lines) preserving index URLs."""
    index_lines: list[str] = []
    package_lines: list[str] = []
    for raw in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--"):
            index_lines.append(stripped)
        elif not stripped.startswith("-r "):
            package_lines.append(stripped)
    return index_lines, package_lines


def flatten_gradio_requirements() -> str:
    index_lines, lines = split_requirement_lines(HF_REQUIREMENTS_FILE)
    for raw in (REPO_ROOT / GRADIO_PIN_FILE).read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith(("-r ", "#")):
            lines.append(stripped)
    return "\n".join(index_lines + lines) + "\n"

# Static gallery: examples/ page flattened to the Space root (index.html
# must sit at root) plus the two small assets it references relatively.
STATIC_README = ("spaces/README-static.md", "README.md")
STATIC_ASSETS = [
    "kokorotts_hf_logo.svg",
    "kokorotts_hf_favicon.svg",
    "kokorotts_hf_bg.svg",
]


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
    files = GRADIO_FILES
    dirs = GRADIO_DIRS
    staged: list[str] = []
    for src, name in files:
        src_path = REPO_ROOT / src
        if not src_path.exists():
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
    if flavour == "gradio":
        (dest / "requirements.txt").write_text(
            flatten_gradio_requirements(), encoding="utf-8"
        )
        sample_dest = dest / GRADIO_SAMPLE_DEST
        sample_dest.mkdir(exist_ok=True)
        for sample in sorted((REPO_ROOT / "examples").glob("kokorotts-*.mp3")):
            shutil.copy2(sample, sample_dest / sample.name)
        staged.append(GRADIO_SAMPLE_DEST + "/")
    return staged


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy KokoroTTS-HF to HF Spaces.")
    parser.add_argument("--flavour", choices=["gradio", "static"], default="gradio")
    parser.add_argument("--space-id", required=True, help="e.g. YOU/kokorotts-hf")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="defaults to $HF_TOKEN")
    parser.add_argument("--dry-run", action="store_true", help="stage files only, no upload")
    parser.add_argument("--stage-dir", default="",
                        help="copy staged Space files into this directory and keep them "
                             "(for manual web drag-drop upload); implies no upload")
    parser.add_argument("--private", action="store_true", help="create a private Space")
    parser.add_argument("--hardware", default="",
                        help="Space hardware, e.g. cpu-basic. "
                             "Empty keeps the account default.")
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
            **({"space_hardware": args.hardware} if args.hardware else {}),
        )
        api.upload_folder(
            repo_id=args.space_id, repo_type="space", folder_path=str(dest),
        )
        print(f"https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
