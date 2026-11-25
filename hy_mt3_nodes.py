"""ComfyUI nodes: an independent Hy-MT3 GGUF loader and a translation node."""

from __future__ import annotations

import os

try:  # package import (ComfyUI)
    from .hy_mt3_backend import (
        HyMT3Model,
        available_gguf_names,
        ensure_model,
        generate,
        register_model_folder,
        resolve_model_path,
    )
    from .hy_mt3_prompt import (
        build_instruction,
        clean_translation,
        prompt_language as detect_prompt_language,
    )
    from .hy_mt3_registry import (
        LANGUAGE_LABELS,
        language_names,
        quantization_for_filename,
        spec_for_filename,
    )
except ImportError:  # direct module import (tests)
    from hy_mt3_backend import (
        HyMT3Model,
        available_gguf_names,
        ensure_model,
        generate,
        register_model_folder,
        resolve_model_path,
    )
    from hy_mt3_prompt import (
        build_instruction,
        clean_translation,
        prompt_language as detect_prompt_language,
    )
    from hy_mt3_registry import (
        LANGUAGE_LABELS,
        language_names,
        quantization_for_filename,
        spec_for_filename,
    )

register_model_folder()

_NGL = {"cpu": 0, "gpu": 999, "auto": -1}
_TASKS = ["default", "terminology", "style", "context", "structured", "custom"]


class HyMT3ModelLoader:
    """Load (and optionally download) a Hy-MT3 GGUF and configure inference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (available_gguf_names(),),
                "auto_download": ("BOOLEAN", {"default": True}),
                "model_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional absolute path. Overrides the model dropdown.",
                    },
                ),
                "backend": (["auto", "cli", "python"],),
                "device": (["auto", "gpu", "cpu"],),
                "ctx_size": ("INT", {"default": 8192, "min": 512, "max": 262144, "step": 512}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "sampling": (["model_recommended", "custom"],),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": -1, "max": 500, "step": 1}),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "timeout": ("INT", {"default": 600, "min": 30, "max": 86400, "step": 30}),
            }
        }

    RETURN_TYPES = ("HY_MT3_MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "load_model"
    CATEGORY = "Hy-MT3"

    def load_model(
        self,
        model: str,
        auto_download: bool,
        model_path: str,
        backend: str,
        device: str,
        ctx_size: int,
        threads: int,
        sampling: str,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        seed: int,
        timeout: int,
    ):
        override = model_path.strip()
        path = resolve_model_path(model, override)
        lookup_name = os.path.basename(path) if override else model
        ensure_model(path, lookup_name, auto_download)

        spec = spec_for_filename(lookup_name)
        if sampling == "model_recommended" and spec is not None:
            temperature, top_p = spec.temperature, spec.top_p
            top_k, repeat_penalty = spec.top_k, spec.repeat_penalty

        handle = HyMT3Model(
            path=path,
            name=lookup_name,
            size=spec.size if spec else "custom",
            quantization=quantization_for_filename(lookup_name),
            repo_id=spec.repo_id if spec else None,
            ctx_size=ctx_size,
            ngl=_NGL.get(device, -1),
            threads=threads,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repeat_penalty=float(repeat_penalty),
            seed=seed,
            timeout=timeout,
            backend=backend,
        )
        info = (
            f"{handle.label} | {path} | ctx={ctx_size} | device={device} | "
            f"sampling={sampling} | temp={handle.temperature} top_p={handle.top_p} "
            f"top_k={handle.top_k} rep={handle.repeat_penalty}"
        )
        print(f"[Hy-MT3] loaded {info}")
        return (handle, info)


class HyMT3Translate:
    """Translate text with a loaded Hy-MT3 model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HY_MT3_MODEL",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "target_language": (LANGUAGE_LABELS,),
                "prompt_language": (["auto", "zh", "en"],),
                "task": (_TASKS,),
                "style": ("STRING", {"default": "", "multiline": False}),
                "glossary": ("STRING", {"default": "", "multiline": True}),
                "background": ("STRING", {"default": "", "multiline": True}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "strip_thinking": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "translate"
    CATEGORY = "Hy-MT3"

    def translate(
        self,
        model: HyMT3Model,
        text: str,
        target_language: str,
        prompt_language: str,
        task: str,
        style: str,
        glossary: str,
        background: str,
        custom_prompt: str,
        max_tokens: int,
        strip_thinking: bool,
    ):
        if not text.strip():
            return ("",)
        target_en, target_zh = language_names(target_language)
        lang = detect_prompt_language(text, prompt_language)
        instruction = build_instruction(
            text=text,
            target_en=target_en,
            target_zh=target_zh,
            lang=lang,
            task=task,
            style=style,
            glossary=glossary,
            background=background,
            custom_prompt=custom_prompt,
        )
        raw = generate(model, instruction, max_tokens, progress=print)
        result = clean_translation(raw) if strip_thinking else raw.strip()
        return (result,)
