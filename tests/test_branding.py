"""Branding regression tests: this fork must look like itself.

Upstream attribution stays where the license requires it (README
attribution, LICENSE, THIRD_PARTY_NOTICES.md, docs history), but all
user-facing surfaces — README hero, gallery page, app header, 404 —
must carry KokoroTTS-HF branding, not upstream's.
"""

import re
import unittest
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[1]
OUR_REPO = "github.com/namng194/kokoroTTS-hf"


def read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


class TestOwnAssets(unittest.TestCase):
    def test_svg_brand_assets_exist_and_parse(self):
        for asset in (
            "assets/kokorotts_hf_favicon.svg",
            "assets/kokorotts_hf_logo.svg",
            "assets/kokorotts_hf_hero.svg",
        ):
            path = REPO_ROOT / asset
            self.assertTrue(path.is_file(), f"missing {asset}")
            root = ElementTree.parse(path).getroot()
            self.assertTrue(root.tag.endswith("svg"), f"{asset} is not SVG")

    def test_images_shipped_where_referenced(self):
        for dockerfile in ("Dockerfile", "Dockerfile.hf"):
            content = read(dockerfile)
            self.assertIn("kokorotts_hf_logo.svg", content, dockerfile)
            self.assertIn("kokorotts_hf_favicon.svg", content, dockerfile)
            self.assertNotIn(".webp", content, f"{dockerfile} ships upstream art")
            self.assertNotIn("hangrylabs", content, dockerfile)

    def test_asset_directory_is_fully_ours(self):
        files = sorted(
            p.name for p in (REPO_ROOT / "assets").iterdir() if p.is_file()
        )
        self.assertTrue(files, "assets/ is empty")
        foreign = [name for name in files if not name.startswith("kokorotts_hf_")]
        self.assertEqual(foreign, [], f"upstream assets remain: {foreign}")


class TestGalleryBranding(unittest.TestCase):
    def test_no_upstream_brand_on_gallery(self):
        for name in ("examples/index.html", "examples/player.js"):
            content = read(name)
            self.assertNotIn("nuggies.website", content, name)
            self.assertNotIn("Hangry Labs", content, name)
            self.assertNotIn("Hangry-Labs", content, name)
            self.assertNotIn("hangrylabs/", content, name)

    def test_gallery_points_at_this_project(self):
        content = read("examples/index.html")
        self.assertIn(OUR_REPO, content)
        self.assertIn("kokorotts_hf_logo.svg", content)
        self.assertIn('#studio', content)

    def test_single_page_inference_covers_all_voices(self):
        content = read("examples/index.html")
        self.assertNotIn("nuggies.website", content)
        self.assertNotIn("Hangry", content)
        self.assertIn("kokoro-js", content)
        self.assertIn("KokoroTTS-HF", content)
        self.assertIn("kokorotts_hf_favicon.svg", content)
        self.assertIn("VOICE_EXAMPLES", content)
        self.assertIn("KokoroStudioUseVoice", content)
        self.assertIn("isLiveVoice", content)
        for voice_id in ("af_heart", "jf_alpha", "diem_trinh", "df_victoria"):
            self.assertIn(voice_id, read("examples/voices.js"))
        self.assertFalse((REPO_ROOT / "examples/studio.html").exists(),
                         "single-page: examples/studio.html must not exist")

    def test_live_demo_key_translated_everywhere(self):
        content = read("examples/player.js")
        self.assertEqual(content.count('liveDemo: "'), 10,
                         "liveDemo key must exist in all 10 locales")
        self.assertNotIn("dockerHub", content)

    def test_i18n_keys_all_defined(self):
        html = read("examples/index.html")
        used = set(re.findall(r'data-i18n="([A-Za-z]+)"', html))
        js = read("examples/player.js")
        en_block = js.split("const TRANSLATIONS = {", 1)[1].split("en: {", 1)[1]
        en_block = en_block.split("},", 1)[0]
        defined = set(re.findall(r"^\s*([A-Za-z]+):", en_block, re.M))
        self.assertTrue(used, "no data-i18n keys found")
        self.assertEqual(used - defined, set(),
                         f"i18n keys used but undefined: {used - defined}")


class TestAppChromeBranding(unittest.TestCase):
    def test_ui_header_is_ours(self):
        content = read("kokorotts/standalone_ui/static/index.html")
        self.assertNotIn("nuggies.website", content)
        self.assertNotIn("hangrylabs_logo", content)
        self.assertNotIn("Hangry-Labs", content)
        self.assertIn("kokorotts_hf_logo.svg", content)
        self.assertIn(OUR_REPO, content)

    def test_404_is_ours(self):
        content = read("404.html")
        self.assertNotIn("nuggies.website", content)
        self.assertNotIn("Hangry Labs", content)
        self.assertIn("KokoroTTS-HF", content)


class TestReadmeBranding(unittest.TestCase):
    def test_hero_is_ours(self):
        content = read("README.md")
        hero = content.split("## Contents")[0]
        self.assertIn("kokorotts_hf_hero.svg", hero)
        self.assertNotIn("nuggies.website", hero)
        self.assertNotIn("kokoro_logo_horizontal.webp", hero)
        self.assertIn("# KokoroTTS-HF", content)

    def test_attribution_preserved(self):
        content = read("README.md")
        self.assertIn("Hangry Labs KokoroTTS", content)
        self.assertIn("hexgrad/Kokoro", content)
        self.assertIn("THIRD_PARTY_NOTICES.md", content)

    def test_support_points_here(self):
        content = read("README.md")
        self.assertIn("github.com/namng194/kokoroTTS-hf/issues", content)


if __name__ == "__main__":
    unittest.main()
