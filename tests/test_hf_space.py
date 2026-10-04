"""Static checks for the Hugging Face Spaces flavour.

Runs with stdlib only: no torch/fastapi/gradio imports, no model downloads,
no GPU. Safe on constrained machines (e.g. 8GB VRAM laptops) and in CI.
"""

import ast
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


class TestDockerfileHf(unittest.TestCase):
    def setUp(self):
        self.content = read("Dockerfile.hf")

    def test_exposes_spaces_port(self):
        self.assertIn("EXPOSE 7860", self.content)

    def test_serves_fastapi_server(self):
        self.assertIn('kokorotts.server', self.content)

    def test_cpu_torch_index(self):
        self.assertIn("download.pytorch.org/whl/cpu", self.content)
        self.assertNotIn("download.pytorch.org/whl/cu130", self.content)

    def test_online_lazy_download(self):
        self.assertRegex(self.content, r"HF_HUB_OFFLINE=0")
        self.assertRegex(self.content, r"TRANSFORMERS_OFFLINE=0")

    def test_persists_under_data(self):
        self.assertIn("/data", self.content)

    def test_cpu_default_device(self):
        self.assertRegex(self.content, r"KOKOROTTS_DEVICE=cpu")

    def test_no_cuda_baked_assets(self):
        # The HF image must stay small: no prefetch of weights at build time.
        self.assertNotIn("prefetch_assets", self.content)


class TestRequirementsHf(unittest.TestCase):
    def setUp(self):
        self.lines = read("requirements-hf.txt").splitlines()

    def test_cpu_index_not_cuda(self):
        blob = "\n".join(self.lines)
        self.assertIn("download.pytorch.org/whl/cpu", blob)
        self.assertNotIn("cu130", blob)

    def test_no_nvidia_stack(self):
        bad = [line for line in self.lines if line.startswith(("nvidia-", "cuda-", "triton=="))]
        self.assertEqual(bad, [], f"CPU-only file must not pin CUDA wheels: {bad}")

    def test_key_pins_match_gpu_requirements(self):
        gpu = read("requirements.txt")
        for package in ("fastapi==", "transformers==", "huggingface-hub==",
                        "numpy==", "uvicorn==", "loguru=="):
            gpu_pin = next((l for l in gpu.splitlines() if l.startswith(package)), None)
            hf_pin = next((l for l in self.lines if l.startswith(package)), None)
            self.assertIsNotNone(gpu_pin, f"missing in requirements.txt: {package}")
            self.assertEqual(hf_pin, gpu_pin, f"version drift for {package}")


class TestGradioApp(unittest.TestCase):
    def test_syntax_and_lazy_runtime(self):
        src = read("spaces/gradio_app.py")
        tree = ast.parse(src)
        # Only module-level imports matter: function-level imports of heavy
        # deps (gradio inside build_demo, kokorotts inside get_runtime) are
        # the lazy pattern we want.
        top_imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.add(node.module.split(".")[0])
        for heavy in ("torch", "gradio", "kokorotts"):
            self.assertNotIn(heavy, top_imports, f"{heavy} must be imported lazily")
        self.assertIn("def generate", src)
        self.assertIn("def get_runtime", src)

    def test_space_frontmatter(self):
        readme = read("spaces/README.md")
        self.assertIn("sdk: gradio", readme)
        self.assertIn("app_file:", readme)
        docker_readme = read("spaces/README-docker.md")
        self.assertIn("sdk: docker", docker_readme)
        self.assertIn("app_port: 7860", docker_readme)


class TestDeployHelper(unittest.TestCase):
    def test_dry_run_docker(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "scripts/deploy_hf_space.py",
                 "--flavour", "docker", "--space-id", "local/kokorotts-hf",
                 "--dry-run"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Dockerfile", proc.stdout)

    def test_dry_run_gradio(self):
        proc = subprocess.run(
            [sys.executable, "scripts/deploy_hf_space.py",
             "--flavour", "gradio", "--space-id", "local/kokorotts-hf-demo",
             "--dry-run"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("app.py", proc.stdout)

    def test_rejects_missing_token_without_dry_run(self):
        env = {k: v for k, v in __import__("os").environ.items() if k != "HF_TOKEN"}
        proc = subprocess.run(
            [sys.executable, "scripts/deploy_hf_space.py",
             "--flavour", "docker", "--space-id", "local/kokorotts-hf"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, env=env,
        )
        self.assertEqual(proc.returncode, 2)


class TestNoCommittedSecrets(unittest.TestCase):
    SECRET = re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})")

    def test_tracked_files_clean(self):
        proc = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0)
        offenders = []
        for rel in proc.stdout.splitlines():
            if not rel.strip():
                continue
            path = REPO_ROOT / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            for match in self.SECRET.finditer(text):
                # Allow variable-name mentions, not values.
                line = text[max(0, match.start() - 80):match.end() + 20]
                if "HF_TOKEN" in line or "GH_TOKEN" in line or "GH_PAT" in line:
                    continue
                offenders.append(f"{rel}: ...{line.strip()}...")
        self.assertEqual(offenders, [], f"possible committed secrets: {offenders}")


if __name__ == "__main__":
    unittest.main()
