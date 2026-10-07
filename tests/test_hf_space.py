"""Static checks for the Hugging Face Spaces bundle.

Runs with stdlib only: no torch/fastapi/gradio imports, no model downloads.
Safe on constrained machines and in CI.
"""

import ast
import os
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

    def test_lean_boot_profile(self):
        self.assertRegex(self.content, r"KOKOROTTS_PRELOAD=standard")
        self.assertRegex(self.content, r"KOKOROTTS_MAX_CHARS=\d+")

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
        # deps (gradio inside build_demo, runtime inside get_runtime) are
        # the lazy pattern we want. kokorotts.catalog/space are light
        # (stdlib-only) and allowed at top level for the voice dropdown.
        # `if TYPE_CHECKING` imports are annotation-only, not runtime deps.
        top_modules = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    top_modules.update(a.name for a in node.names)
                elif node.module:
                    top_modules.add(node.module)
        for heavy in ("torch", "gradio", "kokorotts.runtime", "kokorotts.api"):
            self.assertNotIn(heavy, top_modules, f"{heavy} must be imported lazily")
        self.assertIn("def generate", src)
        self.assertIn("def get_runtime", src)
        self.assertIn("voice_choices", src)
        # CPU contract: no spaces shim, named API for browser clients,
        # families auto-enable on demand.
        self.assertNotIn("@spaces.GPU", src)
        self.assertNotIn("import spaces", src)
        self.assertIn('api_name="generate"', src)
        self.assertIn("set_served_model_families", src)

    def test_production_ui_contract(self):
        src = read("spaces/gradio_app.py")
        # Public API: generate takes text/mode/voices/speed/mix; the signed-in
        # owner profile is injected by Gradio, not passed by clients.
        self.assertIn('api_name="generate"', src)
        self.assertIn(
            "inputs=[text, mode, voice, voice_b, speed, mix]", src
        )
        # Access gate: anonymous guests capped, signed-in owner auto-detected
        # via HF OAuth (no manual key).
        self.assertIn("def is_owner", src)
        self.assertIn("gr.LoginButton", src)
        self.assertIn("gr.OAuthProfile", src)
        self.assertIn("_GUEST_MAX_CHARS = 300", src)
        self.assertIn("KOKOROTTS_OWNER", src)
        self.assertNotIn("AUTHOR_KEY", src)
        self.assertNotIn("author_key", src)
        # Container log records the original text of every request.
        self.assertIn("logger.info", src)
        self.assertIn("user={}", src)
        # Production UX: language filter, char counter, examples, queue.
        self.assertIn("def language_options", src)
        self.assertIn("def voices_for_language", src)
        self.assertIn("Language filter", src)
        self.assertIn("gr.Examples", src)
        self.assertIn("queue(max_size=", src)
        # Vertical layout: inference zone on top, one audition list below
        # (hangry-labs pattern: single list + single player, no 70 players).
        self.assertIn("Tạo giọng đọc", src)
        self.assertIn("Nghe thử giọng mẫu", src)
        self.assertNotIn("gr.Tab", src)
        self.assertNotIn("lang_voices", src)
        self.assertIn("sample_voice = gr.Radio", src)
        self.assertIn("Now auditioning", src)
        self.assertIn("autoplay=True", src)
        self.assertIn("def sample_path_for_voice", src)
        self.assertIn("Nghe thử", src)
        self.assertIn("Be Vietnam Pro", src)
        preview_body = src.split("def preview_voice")[1].split("\ndef ")[0]
        self.assertNotIn("_synthesize", preview_body)
        self.assertNotIn("get_runtime", preview_body)

    def test_space_frontmatter(self):
        readme = read("spaces/README.md")
        self.assertIn("sdk: gradio", readme)
        # OAuth auto-registers the Space so the owner is detected
        # automatically (no manual key).
        self.assertIn("hf_oauth: true", readme)
        # Must match the staged filename (deploy uploads gradio_app.py as app.py).
        self.assertIn("app_file: app.py", readme)

    def test_space_frontmatter_valid_per_hf_rules(self):
        import re
        allowed_colors = {
            "red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray",
        }
        for name in ("spaces/README.md", "spaces/README-static.md"):
            frontmatter = read(name).split("---")[1]
            color = re.search(r"^colorFrom:\s*(\S+)", frontmatter, re.M).group(1)
            self.assertIn(color, allowed_colors, f"{name}: bad colorFrom")
            desc = re.search(r"^short_description:\s*(.+)$", frontmatter, re.M).group(1)
            self.assertLessEqual(len(desc), 60, f"{name}: short_description too long")
            if "static" not in name:
                self.assertIn("sdk: gradio", frontmatter)
            else:
                self.assertIn("sdk: static", frontmatter)


class TestDeployHelper(unittest.TestCase):
    def test_dry_run_gradio(self):
        proc = subprocess.run(
            [sys.executable, "scripts/deploy_hf_space.py",
             "--flavour", "gradio", "--space-id", "local/kokorotts-hf-demo",
             "--dry-run"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("app.py", proc.stdout)
        # pyopenjtalk is source-only: the Space must install build tools.
        self.assertIn("packages.txt", proc.stdout)

    def test_stage_dir_needs_no_token(self):
        import os
        import tempfile
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "scripts/deploy_hf_space.py",
                 "--flavour", "gradio", "--space-id", "local/kokorotts-hf",
                 "--stage-dir", os.path.join(tmp, "space")],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, env=env,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("app.py", proc.stdout)

    def test_staged_requirements_cpu_only(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            staged = os.path.join(tmp, "space")
            proc = subprocess.run(
                [sys.executable, "scripts/deploy_hf_space.py",
                 "--flavour", "gradio", "--space-id", "local/x",
                 "--stage-dir", staged],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            from pathlib import Path
            reqs = (Path(staged) / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.11.0\n", reqs)
        self.assertIn("download.pytorch.org/whl/cpu", reqs)
        self.assertNotIn("spaces==", reqs)
        self.assertIn("gradio[oauth]==", reqs)

    def test_staged_samples_cover_all_voices(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            staged = os.path.join(tmp, "space")
            proc = subprocess.run(
                [sys.executable, "scripts/deploy_hf_space.py",
                 "--flavour", "gradio", "--space-id", "local/x",
                 "--stage-dir", staged],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            from pathlib import Path
            staged_samples = sorted((Path(staged) / "samples").glob("*.mp3"))
            sys.path.insert(0, str(REPO_ROOT))
            from kokorotts.catalog import voice_ids
            expected = {f"kokorotts-{v}.mp3" for v in voice_ids()}
            self.assertEqual({p.name for p in staged_samples}, expected)

    def test_rejects_missing_token_without_dry_run(self):
        env = {k: v for k, v in __import__("os").environ.items() if k != "HF_TOKEN"}
        proc = subprocess.run(
            [sys.executable, "scripts/deploy_hf_space.py",
             "--flavour", "gradio", "--space-id", "local/kokorotts-hf"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, env=env,
        )
        self.assertEqual(proc.returncode, 2)


class TestAuthorGate(unittest.TestCase):
    @staticmethod
    def _module(name):
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            name, str(REPO_ROOT / "spaces" / "gradio_app.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_owner_detected_automatically(self):
        from types import SimpleNamespace
        module = self._module("gapp_gate_owner")
        self.assertTrue(module.is_owner(SimpleNamespace(username="nam194")))
        self.assertFalse(module.is_owner(SimpleNamespace(username="stranger")))
        self.assertFalse(module.is_owner(SimpleNamespace(username=None)))
        self.assertFalse(module.is_owner(None))

    def test_guest_truncated_before_runtime(self):
        module = self._module("gapp_gate_guest")
        calls = {}

        class FakeResult:
            audio = [0.1, 0.2]

        class FakeRuntime:
            def synthesize(self, **kwargs):
                calls.update(kwargs)
                return FakeResult()

        module.get_runtime = lambda: FakeRuntime()
        module.generate("x" * 301, "Single", "af_heart",
                        "af_bella", 1.0, 0.5)
        self.assertEqual(len(calls["text"]), 300)

    def test_no_guest_limit_line_in_ui(self):
        src = read("spaces/gradio_app.py")
        self.assertNotIn("Guests: max", src)

    def test_owner_unlimited_past_runtime_gate(self):
        # Owner with long text must pass validation (only the runtime,
        # not the gate, may stop it — so stub the runtime out).
        from types import SimpleNamespace
        module = self._module("gapp_gate_unlimited")
        calls = {}

        class FakeResult:
            audio = [0.1, 0.2]

        class FakeRuntime:
            def synthesize(self, **kwargs):
                calls.update(kwargs)
                return FakeResult()

        module.get_runtime = lambda: FakeRuntime()
        out = module.generate(
            "y" * 301, "Single", "af_heart", "af_bella", 1.0, 0.5,
            SimpleNamespace(username="nam194"),
        )
        self.assertEqual(out[0], 24000)
        self.assertEqual(len(calls["text"]), 301)


class TestBundledCheckpoints(unittest.TestCase):
    def test_models_cover_all_families(self):
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from kokorotts.catalog import MODEL_FAMILY_CHOICES
        from deploy_hf_space import MODEL_CACHE_DIRS
        from kokorotts import catalog
        repos = set()
        for name in dir(catalog):
            if name.endswith("REPO_ID"):
                repos.add(getattr(catalog, name))
        for family in ("kikiri-german-martin", "kikiri-german-victoria"):
            repos.add(f"kikiri-tts/kikiri-german-{family.split('-')[-1]}")
        cached = {d[len("models--"):].replace("--", "/")
                  for d in MODEL_CACHE_DIRS}
        for repo in ("hexgrad/Kokoro-82M",
                     "kikiri-tts/kikiri-german-martin",
                     "kikiri-tts/kikiri-german-victoria",
                     "contextboxai/Kokoro-Vietnamese"):
            self.assertIn(repo, cached, f"checkpoint not bundled: {repo}")
        self.assertEqual(len(MODEL_FAMILY_CHOICES), 4)

    def test_checkpoint_sourcing(self):
        src = read("spaces/gradio_app.py")
        # Free Space repos cap at 1GB < 1.3GB weights: weights stay on the
        # Hub model repos (lazy fetch), never forced-bundled here.
        self.assertNotIn('KOKOROTTS_BUNDLED_CHECKPOINTS"] = "1"', src)
        # The local-only mechanism still exists for pre-seeded workspaces.
        for name in ("kokorotts/model.py", "kokorotts/pipeline.py",
                     "kokorotts/runtime.py"):
            mod_src = read(name)
            self.assertIn("local_files_only=bundled_only()", mod_src)


class TestVoiceBlendWiring(unittest.TestCase):
    def test_blend_route_and_schema_exist(self):
        api_src = read("kokorotts/api.py")
        self.assertIn('"/tts/blend"', api_src)
        self.assertIn("BlendRequest", api_src)
        self.assertIn("voice_blend", api_src)
        schema_src = read("kokorotts/schemas.py")
        self.assertIn("class BlendRequest", schema_src)
        self.assertIn("voice_b", schema_src)
        runtime_src = read("kokorotts/runtime.py")
        self.assertIn("voice_blend", runtime_src)
        self.assertIn("blend_packs", runtime_src)

    def test_gradio_app_exposes_blend_mode(self):
        src = read("spaces/gradio_app.py")
        self.assertIn("Blend", src)
        self.assertIn("voice_blend", src)


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
