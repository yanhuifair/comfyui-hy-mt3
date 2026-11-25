# ComfyUI Hy-MT3

[GitHub repository](https://github.com/yanhuifair/comfyui-hy-mt3)

ComfyUI nodes for Tencent's **Hy-MT3** translation models in GGUF format: 1.8B, 7B and 30B-A3B (MoE), 102 languages.

[中文说明](README_CN.md)

## Nodes

| Node | Purpose |
| --- | --- |
| **Hy-MT3 Model Loader** | Selects or downloads a GGUF, configures device, context, threads and sampling. Outputs a `HY_MT3_MODEL` handle. |
| **Hy-MT3 Translate** | Takes the model handle and text, builds the official translation prompt and returns the translated string. |

Connect `HY_MT3_MODEL` from the loader to the translate node. One loader can feed any number of translate nodes.

## Supported weights

Official repositories and exact filenames:

| Model | Quantizations | Hugging Face |
| --- | --- | --- |
| Hy-MT3-1.8B | `Q4_K_M`, `Q8_0` | [tencent/Hy-MT3-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) |
| Hy-MT3-7B | `IQ2`, `Q4_K_M`, `Q8_0` | [tencent/Hy-MT3-7B-GGUF](https://huggingface.co/tencent/Hy-MT3-7B-GGUF) |
| Hy-MT3-30B-A3B | `IQ2`, `Q4_K_M`, `Q8_0` | [tencent/Hy-MT3-30B-A3B-GGUF](https://huggingface.co/tencent/Hy-MT3-30B-A3B-GGUF) |

The loader downloads the selected file automatically on first use (`auto_download` on). Set `HF_ENDPOINT` to use a Hugging Face mirror.

## Models stored elsewhere

You are not limited to `ComfyUI/models/hy-mt3/`. Any of these work:

- **Absolute path**: type it into the loader's `model_path` input; it overrides the dropdown. Relative paths that exist are also accepted.
- **Extra directories**: set `HY_MT3_MODEL_PATHS` to one or more directories separated by `:` (`;` on Windows), e.g. `HY_MT3_MODEL_PATHS=/Models/LLM:/Volumes/SSD/gguf`. Every `.gguf` found there appears in the dropdown and is used without copying.
- **ComfyUI `extra_model_paths.yaml`**: add a `hy-mt3` key and restart, e.g.
  ```yaml
  mymodels:
    base_path: /path/to/model/library
    hy-mt3: models/hy-mt3
  ```
- **Existing caches**: `~/.cache/llama.cpp` and Hugging Face snapshot folders for the official `tencent/Hy-MT3-*-GGUF` repos are scanned automatically.

When a selected filename exists in several places, the first match in this order wins: ComfyUI registered `hy-mt3` paths, the default folder, `HY_MT3_MODEL_PATHS`, caches. Use `model_path` to force a specific file.

## Installation

1. Copy this folder into `ComfyUI/custom_nodes/` (already there if you cloned it in place).
2. Install one inference backend:
   - llama.cpp CLI (**preferred**, supports new architectures such as `hy_v3`): `brew install llama.cpp` on macOS, or build from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). `llama-cli` must be in `PATH` or set `HY_MT3_LLAMA_CLI=/path/to/llama-cli`.
   - Alternative: `pip install llama-cpp-python` (use a build with Metal/CUDA if you want GPU offload).
3. Optional: `pip install -r requirements.txt` for faster/resumable downloads via `huggingface_hub`.
4. Restart ComfyUI.

Backend selection in the loader: `auto` (CLI first, then Python), `cli`, or `python`. You can force it globally with `HY_MT3_BACKEND=cli|python`.

## Loader options

- `model`: preset filename or a `.gguf` discovered in any configured model folder.
- `device`: `auto` (llama.cpp decides), `gpu` (offload all layers), `cpu`.
- `ctx_size`: default `8192`, the official context setting; range 512–262144.
- `threads`: 0 = automatic.
- `sampling`: `model_recommended` applies the official settings:
  - 1.8B / 7B: temperature `0.7`, top_p `0.6`, top_k `20`, repetition penalty `1.05`
  - 30B-A3B: temperature `0.7`, top_p `1.0`, top_k `-1` (off), repetition penalty `1.0`
  - `custom` uses the four numeric inputs exposed on the node.
- `seed`: -1 = random. `timeout`: per-call limit in seconds (default 600).

## Translate options

- `text`: source text; multiline and structured text are supported.
- `target_language`: 102 languages, shown as `中文 | Chinese (zh)`. The node sends the full Chinese or English language name required by the model; `prompt_language` chooses the prompt language (`auto` detects CJK in the source).
- `task`:
  - `default` — plain translation, output only.
  - `terminology` — prepends glossary pairs from `glossary` (one `A 翻译成 B` / `A translates to B` per line).
  - `style` — constrains style with `style`.
  - `context` — supplies `background` as context.
  - `structured` — keeps format, indentation and placeholders untranslated.
  - `custom` — uses `custom_prompt`, with `{text}` / `{source_text}` and `{target_lang}` placeholders.
- `max_tokens`: default 1024.
- `strip_thinking`: removes `<think>…</think>` blocks and chat markers from output.

The prompt follows the [official task examples](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF#hy-mt3-translation-task-instruction-examples-chinese-english-comparison), and the GGUF's embedded chat template is used for the CLI backend. The model has no default system prompt.

## Troubleshooting

- **`llama-cli not found`** — install llama.cpp or set `HY_MT3_LLAMA_CLI`.
- **`No Hy-MT3 inference backend found`** — install either backend as described above.
- **Download fails** — check network/proxy, set `HF_ENDPOINT`, or download the exact filename manually into `models/hy-mt3/`.
- **Out of memory** — use a smaller quantization (`IQ2`/`Q4_K_M`), reduce `ctx_size`, or set `device=cpu`.
- **MoE 30B is slow** — use `Q4_K_M`, set `device=gpu`, and raise `threads` for prompt processing if no GPU is available.

<!-- 赞赏码：把图片放到 docs/wechat-pay.png，然后取消下面这行注释
![微信赞赏](docs/wechat-pay.png)
-->
