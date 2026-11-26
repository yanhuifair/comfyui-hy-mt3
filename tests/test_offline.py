"""Offline checks: no model download and no real inference required.

Run with:  python3 tests/test_offline.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("hy_mt3_pkg", ROOT / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules["hy_mt3_pkg"] = package
spec.loader.exec_module(package)

from hy_mt3_pkg.hy_mt3_backend import (  # noqa: E402
    HyMT3Model,
    MODEL_FOLDER,
    available_models,
    generate,
    load_model,
    model_folders,
    resolve_source,
)
import hy_mt3_pkg.hy_mt3_backend as backend_mod  # noqa: E402
from hy_mt3_pkg.hy_mt3_nodes import HyMT3ModelLoader, HyMT3Translate  # noqa: E402
from hy_mt3_pkg.hy_mt3_prompt import (  # noqa: E402
    build_instruction,
    clean_translation,
    prompt_language,
)
from hy_mt3_pkg.hy_mt3_registry import (  # noqa: E402
    LANGUAGES,
    MODEL_CHOICES,
    spec_for_size,
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
        self.assertIn("model_path", loader)
        self.assertIn("dtype", loader)
        self.assertNotIn("backend", loader)
        translate = HyMT3Translate.INPUT_TYPES()["required"]
        self.assertIn("model", translate)
        self.assertIn("text", translate)

    def test_registry(self):
        self.assertEqual(MODEL_CHOICES, ("1.8B", "7B", "30B-A3B"))
        self.assertEqual(len(LANGUAGES), 102)
        self.assertEqual(
            spec_for_size("30B-A3B").repo_id, "tencent/Hy-MT3-30B-A3B"
        )
        self.assertEqual(spec_for_size("7B").top_p, 0.6)
        self.assertEqual(spec_for_size("30B-A3B").top_p, 1.0)

    def test_model_folder(self):
        self.assertEqual(MODEL_FOLDER, "hy-mt3")
        self.assertTrue(model_folders()[0].endswith("models/hy-mt3"))


class TestSourceResolution(unittest.TestCase):
    def test_default_repo(self):
        self.assertEqual(resolve_source("7B"), "tencent/Hy-MT3-7B")

    def test_override_repo_or_path(self):
        self.assertEqual(resolve_source("7B", "org/model"), "org/model")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_source("7B", tmp), tmp)
            file_path = Path(tmp) / "weights.safetensors"
            file_path.write_text("stub", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_source("7B", str(file_path))

    def test_local_choice_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "my-7b"
            folder.mkdir()
            (folder / "config.json").write_text(
                json.dumps({"model_type": "hunyuan_v1_dense", "hidden_size": 4096}),
                encoding="utf-8",
            )
            label = f"7B @ {folder}"
            old = os.environ.get("HY_MT3_MODEL_PATHS")
            os.environ["HY_MT3_MODEL_PATHS"] = tmp
            try:
                self.assertIn(label, available_models())
                self.assertEqual(resolve_source(label), str(folder))
            finally:
                if old is None:
                    os.environ.pop("HY_MT3_MODEL_PATHS", None)
                else:
                    os.environ["HY_MT3_MODEL_PATHS"] = old


class TestPrompts(unittest.TestCase):
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

    def test_no_chat_markers_in_instruction(self):
        out = build_instruction("Hello", "French", "法语", "en")
        self.assertNotIn("<｜hy_User｜>", out)
        self.assertNotIn("<｜hy_Assistant｜>", out)

    def test_terminology_style_context(self):
        term = build_instruction(
            "Hello", "Chinese", "中文", "en", task="terminology", glossary="HOK => 王者荣耀"
        )
        self.assertIn("Reference the following translations", term)
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

    def test_clean_translation(self):
        self.assertEqual(
            clean_translation("<think>thinking</think>\nBonjour<answer></answer>"),
            "Bonjour",
        )
        self.assertEqual(
            clean_translation("Bonjour<｜hy_Assistant｜>"),
            "Bonjour",
        )
        self.assertEqual(
            clean_translation("Bonjour<|hy_Assistant|>"),
            "Bonjour",
        )


class FakeTensor:
    shape = (1, 7)

    def to(self, device):
        self.device = device
        return self


class FakeParam:
    device = "cpu"


class FakeTokenizer:
    pad_token_id = 120002

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        self.messages = messages
        self.add_generation_prompt = add_generation_prompt
        return FakeTensor()

    def decode(self, token_ids, skip_special_tokens=True):
        return "<think>thinking</think>Bonjour le monde"


class FakeModel:
    def parameters(self):
        return iter([FakeParam()])

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 3]]


class TestGenerateWithFakeModel(unittest.TestCase):
    def test_generate_uses_chat_template_and_sampling(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = contextlib.nullcontext
        fake_torch.manual_seed = lambda seed: None
        old_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            tokenizer, model = FakeTokenizer(), FakeModel()
            handle = HyMT3Model(
                source="tencent/Hy-MT3-7B",
                size="7B",
                tokenizer=tokenizer,
                model=model,
                device="cpu",
                dtype="bf16",
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                repeat_penalty=1.05,
                seed=-1,
                ctx_size=8192,
            )
            raw = generate(handle, "Translate Hello", 64)
            self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Translate Hello"}])
            self.assertTrue(tokenizer.add_generation_prompt)
            self.assertEqual(model.kwargs["max_new_tokens"], 64)
            self.assertEqual(model.kwargs["temperature"], 0.7)
            self.assertEqual(model.kwargs["top_k"], 20)
            self.assertEqual(clean_translation(raw), "Bonjour le monde")
        finally:
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch


class TestLoadModelCache(unittest.TestCase):
    """load_model must return fresh handles even when the model is cached."""

    def _fake_imports(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bf16"
        fake_torch.float16 = "fp16"
        fake_torch.float32 = "fp32"
        fake_tf = types.ModuleType("transformers")
        fake_tf.AutoModelForCausalLM = object()
        fake_tf.AutoTokenizer = object()
        return fake_torch, fake_tf

    def test_fresh_handles_no_sampling_leak(self):
        fake_torch, fake_tf = self._fake_imports()
        old_torch, old_tf = sys.modules.get("torch"), sys.modules.get("transformers")
        sys.modules["torch"] = fake_torch
        sys.modules["transformers"] = fake_tf
        prev = dict(backend_mod._CACHE)
        fake_tok = FakeTokenizer()
        fake_model = FakeModel()
        try:
            backend_mod._CACHE[("tencent/Hy-MT3-7B", "bf16", "auto")] = (
                fake_tok,
                fake_model,
                "cpu",
            )
            h1 = load_model("tencent/Hy-MT3-7B", "bf16", "auto")
            h1.temperature = 1.9  # custom override in the loader node
            h1.top_p = 0.99
            h2 = load_model("tencent/Hy-MT3-7B", "bf16", "auto")
            self.assertIsNot(h1, h2)
            self.assertEqual(h2.temperature, 0.7)
            self.assertEqual(h2.top_p, 0.6)
            self.assertEqual(h2.size, "7B")
        finally:
            backend_mod._CACHE.clear()
            backend_mod._CACHE.update(prev)
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch
            if old_tf is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_tf

    def test_local_config_size_and_sampling(self):
        fake_torch, fake_tf = self._fake_imports()
        old_torch, old_tf = sys.modules.get("torch"), sys.modules.get("transformers")
        sys.modules["torch"] = fake_torch
        sys.modules["transformers"] = fake_tf
        prev = dict(backend_mod._CACHE)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / "local-30b"
                folder.mkdir()
                (folder / "config.json").write_text(
                    json.dumps({"model_type": "hy_v3"}), encoding="utf-8"
                )
                fake_tok = FakeTokenizer()
                fake_model = FakeModel()
                backend_mod._CACHE[(str(folder), "bf16", "auto")] = (
                    fake_tok,
                    fake_model,
                    "cpu",
                )
                handle = load_model(str(folder), "bf16", "auto")
                self.assertEqual(handle.size, "30B-A3B")
                self.assertEqual(handle.top_p, 1.0)
                self.assertEqual(handle.top_k, -1)
                self.assertEqual(handle.repeat_penalty, 1.0)
        finally:
            backend_mod._CACHE.clear()
            backend_mod._CACHE.update(prev)
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch
            if old_tf is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_tf


if __name__ == "__main__":
    unittest.main(verbosity=2)
