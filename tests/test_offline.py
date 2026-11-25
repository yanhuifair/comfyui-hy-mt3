"""Offline checks: no model download and no real inference required.

Run with:  python3 tests/test_offline.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the way ComfyUI does (package __init__, with relative submodules).
spec = importlib.util.spec_from_file_location("hy_mt3_pkg", ROOT / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules["hy_mt3_pkg"] = package
spec.loader.exec_module(package)

from hy_mt3_pkg.hy_mt3_backend import MODEL_FOLDER, model_folders  # noqa: E402
from hy_mt3_pkg.hy_mt3_nodes import HyMT3ModelLoader, HyMT3Translate  # noqa: E402
from hy_mt3_pkg.hy_mt3_prompt import (  # noqa: E402
    BOS,
    ASSISTANT,
    USER,
    build_instruction,
    chat_prompt,
    clean_translation,
    prompt_language,
)
from hy_mt3_pkg.hy_mt3_registry import (  # noqa: E402
    LANGUAGES,
    PRESET_FILENAMES,
    spec_for_filename,
)


class TestRegistration(unittest.TestCase):
    def test_nodes_registered(self):
        self.assertEqual(
            set(package.NODE_CLASS_MAPPINGS),
            {"HyMT3ModelLoader", "HyMT3Translate"},
        )
        self.assertEqual(len(package.NODE_DISPLAY_NAME_MAPPINGS), 2)

    def test_input_types(self):
        loader = HyMT3ModelLoader.INPUT_TYPES()["required"]
        self.assertIn("model", loader)
        self.assertIn("auto_download", loader)
        self.assertIn("ctx_size", loader)
        translate = HyMT3Translate.INPUT_TYPES()["required"]
        self.assertIn("model", translate)
        self.assertIn("text", translate)
        self.assertEqual(len(HyMT3Translate.RETURN_TYPES), 1)

    def test_model_folder(self):
        self.assertEqual(MODEL_FOLDER, "hy-mt3")
        self.assertTrue(model_folders()[0].endswith("models/hy-mt3"))

    def test_registry(self):
        self.assertEqual(len(PRESET_FILENAMES), 8)
        self.assertEqual(len(LANGUAGES), 102)
        self.assertEqual(len(set(PRESET_FILENAMES)), 8)
        for name in PRESET_FILENAMES:
            self.assertIsNotNone(spec_for_filename(name), name)


class TestPrompts(unittest.TestCase):
    def test_chat_template(self):
        self.assertEqual(
            chat_prompt("hello"),
            f"{BOS}{USER}hello{ASSISTANT}",
        )

    def test_prompt_language_detection(self):
        self.assertEqual(prompt_language("今天天气真好", "auto"), "zh")
        self.assertEqual(prompt_language("Hello world", "auto"), "en")
        self.assertEqual(prompt_language("你好", "en"), "en")

    def test_default_prompts(self):
        zh = build_instruction("今天天气真好", "English", "英语", "zh")
        self.assertIn("翻译为英语", zh)
        self.assertIn("今天天气真好", zh)
        en = build_instruction("Hello", "French", "法语", "en")
        self.assertIn("Translate the following text into French", en)

    def test_terminology_style_context(self):
        term = build_instruction(
            "Hello", "Chinese", "中文", "en", task="terminology", glossary="HOK => 王者荣耀"
        )
        self.assertIn("Reference the following translations", term)
        self.assertIn("HOK => 王者荣耀", term)
        style = build_instruction("Hello", "Chinese", "中文", "en", task="style", style="poetic")
        self.assertIn("strictly conform to [**poetic**]", style)
        ctx = build_instruction(
            "Hello", "Chinese", "中文", "en", task="context", background="a game"
        )
        self.assertIn("[Background Information]\na game", ctx)

    def test_custom_prompt_placeholders(self):
        out = build_instruction(
            "Hello",
            "Japanese",
            "日语",
            "en",
            task="custom",
            custom_prompt="Translate {text} into {target_lang}, no notes.",
        )
        self.assertEqual(out, "Translate Hello into Japanese, no notes.")

    def test_structured_prompt_literal_placeholders(self):
        out = build_instruction("{\"k\": \"你好\"}", "English", "英语", "zh", task="structured")
        self.assertIn("{{var}}", out)
        self.assertIn("${var}", out)

    def test_clean_translation(self):
        self.assertEqual(
            clean_translation("<think>thinking</think>\nBonjour<answer></answer>"),
            "Bonjour",
        )
        self.assertEqual(clean_translation("  Hello  "), "Hello")


class TestEndToEndWithStubCLI(unittest.TestCase):
    def test_loader_to_translate(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "Hy-MT3-1.8B-Q4_K_M.gguf"
            model_file.write_bytes(b"not a real gguf")
            stub = Path(tmp) / "llama-cli"
            stub.write_text("#!/bin/sh\nprintf '%s\\n' '<think>thinking</think>Bonjour le monde'\n")
            stub.chmod(0o755)

            old_cli = os.environ.get("HY_MT3_LLAMA_CLI")
            old_backend = os.environ.get("HY_MT3_BACKEND")
            os.environ["HY_MT3_LLAMA_CLI"] = str(stub)
            os.environ.pop("HY_MT3_BACKEND", None)
            try:
                loaded = HyMT3ModelLoader().load_model(
                    model="Hy-MT3-1.8B-Q4_K_M.gguf",
                    auto_download=False,
                    model_path=str(model_file),
                    backend="auto",
                    device="cpu",
                    ctx_size=8192,
                    threads=0,
                    sampling="model_recommended",
                    temperature=0.0,
                    top_p=0.0,
                    top_k=0,
                    repeat_penalty=1.0,
                    seed=-1,
                    timeout=30,
                )
                handle = loaded[0]
                self.assertEqual(handle.temperature, 0.7)
                self.assertEqual(handle.top_p, 0.6)
                self.assertEqual(handle.top_k, 20)
                self.assertEqual(handle.ngl, 0)

                result = HyMT3Translate().translate(
                    model=handle,
                    text="Hello",
                    target_language="法语 | French (fr)",
                    prompt_language="auto",
                    task="default",
                    style="",
                    glossary="",
                    background="",
                    custom_prompt="",
                    max_tokens=64,
                    strip_thinking=True,
                )
                self.assertEqual(result[0], "Bonjour le monde")
            finally:
                if old_cli is None:
                    os.environ.pop("HY_MT3_LLAMA_CLI", None)
                else:
                    os.environ["HY_MT3_LLAMA_CLI"] = old_cli
                if old_backend is not None:
                    os.environ["HY_MT3_BACKEND"] = old_backend


if __name__ == "__main__":
    unittest.main(verbosity=2)
