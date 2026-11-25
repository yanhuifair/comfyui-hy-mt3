"""ComfyUI nodes: an independent Hy-MT3 loader and a translation node."""

from __future__ import annotations

try:  # package import (ComfyUI)
    from .hy_mt3_backend import (
        available_models,
        generate,
        load_model,
        register_model_folder,
        resolve_source,
    )
    from .hy_mt3_prompt import (
        build_instruction,
        clean_translation,
        prompt_language as detect_prompt_language,
    )
    from .hy_mt3_registry import LANGUAGE_LABELS, language_names
except ImportError:  # direct module import (tests)
    from hy_mt3_backend import (
        available_models,
        generate,
        load_model,
        register_model_folder,
        resolve_source,
    )
    from hy_mt3_prompt import (
        build_instruction,
        clean_translation,
        prompt_language as detect_prompt_language,
    )
    from hy_mt3_registry import LANGUAGE_LABELS, language_names

register_model_folder()

_TASKS = ["default", "terminology", "style", "context", "structured", "custom"]


class HyMT3ModelLoader:
    """Load a full-weight Hy-MT3 model from Hugging Face or a local folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (available_models(),),
                "model_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Local model folder or HF repo id. Overrides the dropdown.",
                    },
                ),
                "device": (["auto", "cuda", "mps", "cpu"],),
                "dtype": (["bf16", "fp16", "fp32"],),
                "sampling": (["model_recommended", "custom"],),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": -1, "max": 500, "step": 1}),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("HY_MT3_MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "load_model"
    CATEGORY = "Hy-MT3"

    def load_model(
        self,
        model: str,
        model_path: str,
        device: str,
        dtype: str,
        sampling: str,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        seed: int,
    ):
        source = resolve_source(model, model_path)
        handle = load_model(source, dtype=dtype, device=device, progress=print)
        if sampling == "custom":
            handle.temperature = float(temperature)
            handle.top_p = float(top_p)
            handle.top_k = int(top_k)
            handle.repeat_penalty = float(repeat_penalty)
        handle.seed = seed

        info = (
            f"{handle.label} | source={source} | sampling={sampling} | "
            f"temp={handle.temperature} top_p={handle.top_p} "
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
        model,
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
