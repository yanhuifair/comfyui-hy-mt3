"""Prompt construction for Hy-MT3.

The prompts below follow the official model card examples:
https://huggingface.co/tencent/Hy-MT3-1.8B#hy-mt3-translation-task-instruction-examples-chinese-english-comparison

The model has no default system prompt. Chat markers are deliberately NOT
added here: the backend uses ``tokenizer.apply_chat_template`` so each model
uses its own template (7B's differs from 1.8B and 30B-A3B).
"""

from __future__ import annotations

import re


def prompt_language(text: str, preference: str) -> str:
    if preference == "zh":
        return "zh"
    if preference == "en":
        return "en"
    return "zh" if re.search(r"[\u3400-\u9fff]", text) else "en"


def _default_zh(target: str, text: str) -> str:
    return (
        f"将以下文本翻译为{target}，注意**只需要输出翻译后的结果，不要额外解释**：\n\n{text}"
    )


def _default_en(target: str, text: str) -> str:
    return (
        f"Translate the following text into {target}. Note that you should "
        f"**only output the translated result without any additional explanation**:\n\n{text}"
    )


def _glossary_block(pairs: str, lang: str) -> str:
    lines = [line.strip() for line in pairs.splitlines() if line.strip()]
    if not lines:
        return ""
    if lang == "zh":
        return "_参考下面的翻译：_\n" + "\n".join(lines) + "\n\n"
    return "_Reference the following translations:_\n" + "\n".join(lines) + "\n\n"


def build_instruction(
    text: str,
    target_en: str,
    target_zh: str,
    lang: str,
    task: str = "default",
    style: str = "",
    glossary: str = "",
    background: str = "",
    custom_prompt: str = "",
) -> str:
    """Build the user message consumed by the model (without chat markers)."""
    target = target_zh if lang == "zh" else target_en

    if task == "custom":
        template = custom_prompt.strip()
        if not template:
            raise ValueError("Custom mode requires a prompt template.")
        if "{text}" not in template and "{source_text}" not in template:
            template = f"{template}\n\n{text}"
        return (
            template.replace("{source_text}", text)
            .replace("{text}", text)
            .replace("{target_lang}", target)
        )

    if task == "terminology":
        prefix = _glossary_block(glossary, lang)
        return prefix + (_default_zh(target, text) if lang == "zh" else _default_en(target, text))

    if task == "style":
        if lang == "zh":
            return (
                f"请将以下文本翻译为{target}。\n注意翻译的风格要严格符合【**{style}**】\n\n{text}"
            )
        return (
            f"Please translate the following text into {target}. Note that the "
            f"translation style must strictly conform to [**{style}**]:\n\n{text}"
        )

    if task == "context":
        if lang == "zh":
            return (
                f"【背景信息】\n{background}\n\n请结合背景信息将以下文本翻译为{target}。\n\n"
                f"【待翻译文本】\n{text}"
            )
        return (
            f"[Background Information]\n{background}\n\nPlease translate the following "
            f"text into {target}, taking the provided background information into "
            f"consideration.\n\n[Source Text]\n{text}"
        )

    if task == "structured":
        if lang == "zh":
            return (
                f"# 任务目标\n将下方文本中的可见文本翻译为{target}。\n\n"
                f"# 严格约束\n1. **结构锁定**：保持原有格式、缩进和层级完全不变。\n"
                f"2. **选择性翻译**：仅翻译面向用户展示的可见文本。\n"
                f"3. **禁止修改**：严禁翻译或更改任何代码标签、键名、变量占位符"
                f"（如 {{{{var}}}}、${{var}}、%s、%d 等）或代码属性。\n\n"
                f"# 数据输入\n{text}"
            )
        return (
            f"### Task\nTranslate the user-facing text within the following data into "
            f"{target}.\n\n### Strict Rules\n1. **Structure Preservation:** Preserve the "
            f"original format, nesting, hierarchy and indentation exactly.\n"
            f"2. **Selective Translation:** Translate ONLY visible, user-facing text.\n"
            f"3. **Strict Non-Translation:** NEVER translate or alter code tags, keys, "
            f"placeholders (e.g. {{{{var}}}}, ${{var}}, %s, %d) or properties.\n\n"
            f"### Source Data\n{text}"
        )

    return _default_zh(target, text) if lang == "zh" else _default_en(target, text)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# Hy-MT3 markers appear in fullwidth form (<｜hy_Assistant｜>); match both.
_MARKERS = re.compile(
    r"</?think>|</?answer>|<[|｜]hy_Assistant[|｜]>|<[|｜]hy_end▁of▁sentence[|｜]>",
    re.IGNORECASE,
)


def clean_translation(raw: str) -> str:
    """Strip reasoning blocks and chat markers from the model output."""
    text = _THINK_BLOCK.sub("", raw or "")
    text = _MARKERS.sub("", text)
    text = text.replace("<eos:6124c78e>", "")
    return text.strip()
