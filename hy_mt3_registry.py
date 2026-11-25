"""Model registry and language table for the Hy-MT3 ComfyUI nodes.

The three official full-weight (BF16 safetensors) repositories:
  https://huggingface.co/tencent/Hy-MT3-1.8B
  https://huggingface.co/tencent/Hy-MT3-7B
  https://huggingface.co/tencent/Hy-MT3-30B-A3B
"""

from __future__ import annotations

from dataclasses import dataclass

HF_BASE_URL = "https://huggingface.co"


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    size: str
    architecture: str
    params_b: float
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float


# Official recommended sampling parameters from the model cards.
# 1.8B / 7B:   temp 0.7, top_p 0.6, top_k 20,  repetition_penalty 1.05
# 30B-A3B:     temp 0.7, top_p 1.0, top_k -1,  repetition_penalty 1.0
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        repo_id="tencent/Hy-MT3-1.8B",
        size="1.8B",
        architecture="HunYuanDenseV1ForCausalLM",
        params_b=1.79,
        temperature=0.7,
        top_p=0.6,
        top_k=20,
        repeat_penalty=1.05,
    ),
    ModelSpec(
        repo_id="tencent/Hy-MT3-7B",
        size="7B",
        architecture="HunYuanDenseV1ForCausalLM",
        params_b=7.50,
        temperature=0.7,
        top_p=0.6,
        top_k=20,
        repeat_penalty=1.05,
    ),
    ModelSpec(
        repo_id="tencent/Hy-MT3-30B-A3B",
        size="30B-A3B",
        architecture="HYV3ForCausalLM",
        params_b=30.06,
        temperature=0.7,
        top_p=1.0,
        top_k=-1,
        repeat_penalty=1.0,
    ),
)

MODEL_CHOICES: tuple[str, ...] = tuple(spec.size for spec in MODEL_SPECS)


def spec_for_size(size: str) -> ModelSpec | None:
    for spec in MODEL_SPECS:
        if spec.size == size:
            return spec
    return None


def spec_for_repo(repo_id: str) -> ModelSpec | None:
    for spec in MODEL_SPECS:
        if spec.repo_id == repo_id:
            return spec
    return None


# 102 languages supported by Hy-MT3. Each entry:
# (combo label, English full name, Chinese full name).
# The model card requires full language names: Chinese names for Chinese
# prompts, English names for English prompts.
LANGUAGES: tuple[tuple[str, str, str], ...] = (
    ("中文 | Chinese (zh)", "Chinese", "中文"),
    ("英语 | English (en)", "English", "英语"),
    ("法语 | French (fr)", "French", "法语"),
    ("葡萄牙语 | Portuguese (pt)", "Portuguese", "葡萄牙语"),
    ("西班牙语 | Spanish (es)", "Spanish", "西班牙语"),
    ("日语 | Japanese (ja)", "Japanese", "日语"),
    ("土耳其语 | Turkish (tr)", "Turkish", "土耳其语"),
    ("俄语 | Russian (ru)", "Russian", "俄语"),
    ("阿拉伯语 | Arabic (ar)", "Arabic", "阿拉伯语"),
    ("韩语 | Korean (ko)", "Korean", "韩语"),
    ("泰语 | Thai (th)", "Thai", "泰语"),
    ("越南语 | Vietnamese (vi)", "Vietnamese", "越南语"),
    ("马来语 | Malay (ms)", "Malay", "马来语"),
    ("印尼语 | Indonesian (id)", "Indonesian", "印尼语"),
    ("菲律宾语 | Filipino (tl)", "Filipino", "菲律宾语"),
    ("印地语 | Hindi (hi)", "Hindi", "印地语"),
    ("马拉地语 | Marathi (mr)", "Marathi", "马拉地语"),
    ("泰卢固语 | Telugu (te)", "Telugu", "泰卢固语"),
    ("乌尔都语 | Urdu (ur)", "Urdu", "乌尔都语"),
    ("古吉拉特语 | Gujarati (gu)", "Gujarati", "古吉拉特语"),
    ("波斯语 | Persian (fa)", "Persian", "波斯语"),
    ("希伯来语 | Hebrew (he)", "Hebrew", "希伯来语"),
    ("波兰语 | Polish (pl)", "Polish", "波兰语"),
    ("孟加拉语 | Bengali (bn)", "Bengali", "孟加拉语"),
    ("泰米尔语 | Tamil (ta)", "Tamil", "泰米尔语"),
    ("乌克兰语 | Ukrainian (uk)", "Ukrainian", "乌克兰语"),
    ("捷克语 | Czech (cs)", "Czech", "捷克语"),
    ("荷兰语 | Dutch (nl)", "Dutch", "荷兰语"),
    ("意大利语 | Italian (it)", "Italian", "意大利语"),
    ("德语 | German (de)", "German", "德语"),
    ("高棉语 | Khmer (km)", "Khmer", "高棉语"),
    ("缅甸语 | Burmese (my)", "Burmese", "缅甸语"),
    ("繁体中文 | Traditional Chinese (zh-TR)", "Traditional Chinese", "繁体中文"),
    ("藏语 | Tibetan (bo)", "Tibetan", "藏语"),
    ("传统蒙古语 | Traditional Mongolian (mn)", "Traditional Mongolian", "传统蒙古语"),
    ("粤语 | Cantonese (yue)", "Cantonese", "粤语"),
    ("阿拉伯字母哈萨克语 | Arabic Kazakh (kk)", "Arabic Kazakh", "阿拉伯字母哈萨克语"),
    ("维吾尔语 | Uyghur (ug)", "Uyghur", "维吾尔语"),
    ("希腊语 | Greek (el)", "Greek", "希腊语"),
    ("威尔士语 | Welsh (cy)", "Welsh", "威尔士语"),
    ("僧伽罗语 | Sinhala (si)", "Sinhala", "僧伽罗语"),
    ("罗马尼亚语 | Romanian (ro)", "Romanian", "罗马尼亚语"),
    ("加泰罗尼亚语 | Catalan (ca)", "Catalan", "加泰罗尼亚语"),
    ("旁遮普语 | Punjabi (pa)", "Punjabi", "旁遮普语"),
    ("匈牙利语 | Hungarian (hu)", "Hungarian", "匈牙利语"),
    ("加利西亚语 | Galician (gl)", "Galician", "加利西亚语"),
    ("信德语 | Sindhi (sd)", "Sindhi", "信德语"),
    ("保加利亚语 | Bulgarian (bg)", "Bulgarian", "保加利亚语"),
    ("巴斯克语 | Basque (eu)", "Basque", "巴斯克语"),
    ("阿萨姆语 | Assamese (as)", "Assamese", "阿萨姆语"),
    ("塞尔维亚语 | Serbian (sr)", "Serbian", "塞尔维亚语"),
    ("南非荷兰语 | Afrikaans (af)", "Afrikaans", "南非荷兰语"),
    ("奥里亚语 | Odia (or)", "Odia", "奥里亚语"),
    ("克罗地亚语 | Croatian (hr)", "Croatian", "克罗地亚语"),
    ("卢森堡语 | Luxembourgish (lb)", "Luxemburgish", "卢森堡语"),
    ("卡纳达语 | Kannada (kn)", "Kannada", "卡纳达语"),
    ("斯洛伐克语 | Slovak (sk)", "Slovak", "斯洛伐克语"),
    ("马耳他语 | Maltese (mt)", "Maltese", "马耳他语"),
    ("马拉雅拉姆语 | Malayalam (ml)", "Malayalam", "马拉雅拉姆语"),
    ("斯洛文尼亚语 | Slovenian (sl)", "Slovenian", "斯洛文尼亚语"),
    ("世界语 | Esperanto (eo)", "Esperanto", "世界语"),
    ("梵语 | Sanskrit (sa)", "Sanskrit", "梵语"),
    ("阿尔巴尼亚语 | Albanian (sq)", "Albanian", "阿尔巴尼亚语"),
    ("意第绪语 | Yiddish (yi)", "Yiddish", "意第绪语"),
    ("老挝语 | Lao (lo)", "Lao", "老挝语"),
    ("马其顿语 | Macedonian (mk)", "Macedonian", "马其顿语"),
    ("亚美尼亚语 | Armenian (hy)", "Armenian", "亚美尼亚语"),
    ("爪哇语 | Javanese (jv)", "Javanese", "爪哇语"),
    ("白俄罗斯语 | Belarusian (be)", "Belarusian", "白俄罗斯语"),
    ("格鲁吉亚语 | Georgian (ka)", "Georgian", "格鲁吉亚语"),
    ("巽他语 | Sundanese (su)", "Sundanese", "巽他语"),
    ("立陶宛语 | Lithuanian (lt)", "Lithuanian", "立陶宛语"),
    ("阿塞拜疆语 | Azerbaijani (az)", "Azerbaijani", "阿塞拜疆语"),
    ("宿务语 | Cebuano (ceb)", "Cebuano", "宿务语"),
    ("拉脱维亚语 | Latvian (lv)", "Latvian", "拉脱维亚语"),
    ("乌兹别克语 | Uzbek (uz)", "Uzbek", "乌兹别克语"),
    ("瓦瑞语 | Waray (war)", "Waray", "瓦瑞语"),
    ("爱沙尼亚语 | Estonian (et)", "Estonian", "爱沙尼亚语"),
    ("吉尔吉斯语 | Kyrgyz (ky)", "Kyrgyz", "吉尔吉斯语"),
    ("斯瓦希里语 | Swahili (sw)", "Swahili", "斯瓦希里语"),
    ("芬兰语 | Finnish (fi)", "Finnish", "芬兰语"),
    ("土库曼语 | Turkmen (tk)", "Turkmen", "土库曼语"),
    ("阿姆哈拉语 | Amharic (am)", "Amharic", "阿姆哈拉语"),
    ("瑞典语 | Swedish (sv)", "Swedish", "瑞典语"),
    ("塔吉克语 | Tajik (tg)", "Tajik", "塔吉克语"),
    ("索马里语 | Somali (so)", "Somali", "索马里语"),
    ("丹麦语 | Danish (da)", "Danish", "丹麦语"),
    ("鞑靼语 | Tatar (tt)", "Tatar", "鞑靼语"),
    ("马达加斯加语 | Malagasy (mg)", "Malagasy", "马达加斯加语"),
    ("挪威语 | Norwegian (nb)", "Norwegian", "挪威语"),
    ("巴什基尔语 | Bashkir (ba)", "Bashkir", "巴什基尔语"),
    ("海地克里奥尔语 | Haitian Creole (ht)", "Haitian Creole", "海地克里奥尔语"),
    ("新挪威语 | Norwegian Nynorsk (nn)", "Norwegian Nynorsk", "新挪威语"),
    ("库尔德语 | Kurdish (ku)", "Kurdish", "库尔德语"),
    ("西里尔哈萨克语 | Cyrillic Kazakh (kk-Cyrl)", "Cyrillic Kazakh", "西里尔哈萨克语"),
    ("冰岛语 | Icelandic (is)", "Icelandic", "冰岛语"),
    ("中库尔德语 | Central Kurdish (ckb)", "Central Kurdish", "中库尔德语"),
    ("西里尔蒙古语 | Cyrillic Mongolian (mn-Cyrl)", "Cyrillic Mongolian", "西里尔蒙古语"),
    ("爱尔兰语 | Irish (ga)", "Irish", "爱尔兰语"),
    ("普什图语 | Pashto (ps)", "Pashto", "普什图语"),
    ("苏格兰盖尔语 | Scottish Gaelic (gd)", "Scottish Gaelic", "苏格兰盖尔语"),
    ("尼泊尔语 | Nepali (ne)", "Nepali", "尼泊尔语"),
)

LANGUAGE_LABELS: tuple[str, ...] = tuple(label for label, _, _ in LANGUAGES)
_LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    label: (en, zh) for label, en, zh in LANGUAGES
}


def language_names(label: str) -> tuple[str, str]:
    """Return (English name, Chinese name) for a combo label."""
    return _LANGUAGE_MAP.get(label, (label, label))
