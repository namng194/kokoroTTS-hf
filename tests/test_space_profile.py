"""Unit tests for the constrained-host profiles (kokorotts.space).

Stdlib-only: no torch, no model downloads, no GPU. Safe on 8GB VRAM
laptops and in CI.
"""

import os
import unittest
from unittest.mock import patch

from kokorotts import space
from kokorotts.catalog import STANDARD_MODEL_FAMILY


def clean_env(*names: str) -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in names}


PROFILE_VARS = (
    "KOKOROTTS_PRELOAD",
    "KOKOROTTS_MAX_CHARS",
    "KOKOROTTS_SERVED_MODEL_FAMILIES",
    "KOKOROTTS_SERVED_VOICES",
    "KOKOROTTS_HF_SPACE",
    "SPACE_ID",
)


class TestPreloadMode(unittest.TestCase):
    def test_default_is_all(self):
        with patch.dict(os.environ, clean_env(*PROFILE_VARS), clear=True):
            self.assertEqual(space.resolve_preload_mode(), "all")

    def test_standard_and_lazy_accepted(self):
        for mode in ("standard", "lazy"):
            with patch.dict(os.environ, {"KOKOROTTS_PRELOAD": mode}):
                self.assertEqual(space.resolve_preload_mode(), mode)

    def test_unknown_mode_rejected(self):
        with patch.dict(os.environ, {"KOKOROTTS_PRELOAD": "turbo"}):
            with self.assertRaises(ValueError):
                space.resolve_preload_mode()

    def test_build_runtime_kwargs_default_is_eager(self):
        with patch.dict(os.environ, clean_env(*PROFILE_VARS), clear=True):
            self.assertEqual(space.build_runtime_kwargs(), {"eager_voices": True})

    def test_build_runtime_kwargs_lazy_disables_eager(self):
        with patch.dict(os.environ, {"KOKOROTTS_PRELOAD": "lazy"}):
            self.assertEqual(space.build_runtime_kwargs(), {"eager_voices": False})

    def test_standard_seeds_family_default_without_clobbering(self):
        base = clean_env(*PROFILE_VARS)
        with patch.dict(os.environ, {**base, "KOKOROTTS_PRELOAD": "standard"}, clear=True):
            space.apply_preload_defaults()
            self.assertEqual(
                os.environ.get("KOKOROTTS_SERVED_MODEL_FAMILIES"),
                STANDARD_MODEL_FAMILY,
            )
        with patch.dict(
            os.environ,
            {"KOKOROTTS_PRELOAD": "standard", "KOKOROTTS_SERVED_VOICES": "af_heart"},
        ):
            space.apply_preload_defaults()
            self.assertNotIn("KOKOROTTS_SERVED_MODEL_FAMILIES", os.environ)


class TestMaxChars(unittest.TestCase):
    def test_default_unlimited(self):
        with patch.dict(os.environ, clean_env(*PROFILE_VARS), clear=True):
            self.assertIsNone(space.resolve_max_chars())
            self.assertEqual(space.check_text_length(" hello "), "hello")

    def test_limit_enforced(self):
        with patch.dict(os.environ, {"KOKOROTTS_MAX_CHARS": "10"}):
            self.assertEqual(space.resolve_max_chars(), 10)
            self.assertEqual(space.check_text_length("1234567890"), "1234567890")
            with self.assertRaises(ValueError):
                space.check_text_length("12345678901")

    def test_invalid_limit_rejected(self):
        for raw in ("0", "-5", "abc"):
            with patch.dict(os.environ, {"KOKOROTTS_MAX_CHARS": raw}):
                with self.assertRaises(ValueError):
                    space.resolve_max_chars()

    def test_explicit_limit_argument(self):
        self.assertEqual(space.check_text_length("abc", limit=3), "abc")
        with self.assertRaises(ValueError):
            space.check_text_length("abcd", limit=3)


class TestSpaceEnv(unittest.TestCase):
    def test_space_detection(self):
        with patch.dict(os.environ, clean_env(*PROFILE_VARS), clear=True):
            self.assertFalse(space.is_space_env())
        with patch.dict(os.environ, {"SPACE_ID": "user/demo"}):
            self.assertTrue(space.is_space_env())
        with patch.dict(os.environ, clean_env(*PROFILE_VARS), clear=True):
            with patch.dict(os.environ, {"KOKOROTTS_HF_SPACE": "1"}):
                self.assertTrue(space.is_space_env())

    def test_profile_summary_shape(self):
        with patch.dict(os.environ, clean_env(*PROFILE_VARS), clear=True):
            summary = space.profile_summary()
            self.assertEqual(
                summary,
                {
                    "space_env": False,
                    "preload": "all",
                    "max_chars": None,
                    "served_model_families_env": "",
                    "served_voices_env": "",
                },
            )


if __name__ == "__main__":
    unittest.main()
