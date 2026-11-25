# ComfyUI Hy-MT3

[GitHub 仓库](https://github.com/yanhuifair/comfyui-hy-mt3)

面向腾讯 **Hy-MT3** GGUF 翻译模型的 ComfyUI 节点，支持 1.8B、7B、30B-A3B（MoE）三种规格，覆盖 102 种语言。

[English README](README.md)

## 节点

| 节点 | 作用 |
| --- | --- |
| **Hy-MT3 Model Loader** | 选择或下载 GGUF，配置设备、上下文、线程与采样参数，输出 `HY_MT3_MODEL` 句柄。 |
| **Hy-MT3 Translate** | 接收模型句柄与文本，按官方提示词模板翻译并返回译文字符串。 |

把加载器的 `HY_MT3_MODEL` 连到翻译节点；一个加载器可以驱动多个翻译节点。

## 支持的权重

官方仓库与准确文件名：

| 模型 | 量化 | Hugging Face |
| --- | --- | --- |
| Hy-MT3-1.8B | `Q4_K_M`、`Q8_0` | [tencent/Hy-MT3-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) |
| Hy-MT3-7B | `IQ2`、`Q4_K_M`、`Q8_0` | [tencent/Hy-MT3-7B-GGUF](https://huggingface.co/tencent/Hy-MT3-7B-GGUF) |
| Hy-MT3-30B-A3B | `IQ2`、`Q4_K_M`、`Q8_0` | [tencent/Hy-MT3-30B-A3B-GGUF](https://huggingface.co/tencent/Hy-MT3-30B-A3B-GGUF) |

首次运行会自动下载所选文件（`auto_download` 开启时）。也可以把任意 `.gguf` 放到 `ComfyUI/models/hy_mt3/` 后在下拉框中选择，或在 `model_path` 填绝对路径。可用 `HF_ENDPOINT` 指定镜像站。

## 安装

1. 把本目录放到 `ComfyUI/custom_nodes/`（若已克隆到此处则无需移动）。
2. 安装任一推理后端：
   - llama.cpp 命令行（**推荐**，能第一时间支持 `hy_v3` 等新架构）：macOS 用 `brew install llama.cpp`，其它平台参考 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 编译。`llama-cli` 需在 `PATH` 中，或用 `HY_MT3_LLAMA_CLI=/path/to/llama-cli` 指定。
   - 备选：`pip install llama-cpp-python`（需要 GPU 时安装带 Metal/CUDA 的构建）。
3. 可选：`pip install -r requirements.txt`，通过 `huggingface_hub` 获得更快、可续传的下载。
4. 重启 ComfyUI。

加载器中的后端选项：`auto`（优先 CLI，其次 Python）、`cli`、`python`；也可用 `HY_MT3_BACKEND=cli|python` 全局强制。

## 加载器参数

- `model`：预设文件名，或在 `models/hy_mt3/` 中发现的本地 `.gguf`。
- `device`：`auto`（由 llama.cpp 决定）、`gpu`（全部层卸载到显卡）、`cpu`。
- `ctx_size`：默认 `8192`，与官方设置一致；范围 512–262144。
- `threads`：0 表示自动。
- `sampling`：`model_recommended` 使用官方推荐值：
  - 1.8B / 7B：温度 `0.7`、top_p `0.6`、top_k `20`、重复惩罚 `1.05`
  - 30B-A3B：温度 `0.7`、top_p `1.0`、top_k `-1`（关闭）、重复惩罚 `1.0`
  - 选 `custom` 时使用节点上暴露的四个数值输入。
- `seed`：-1 表示随机；`timeout`：单次调用超时秒数，默认 600。

## 翻译节点参数

- `text`：待翻译文本，支持多行与结构化文本。
- `target_language`：102 种语言，显示为 `中文 | Chinese (zh)`。节点会按模型要求传入完整的中文或英文语言名；`prompt_language` 控制提示词语言（`auto` 根据源文是否含中文字符判断）。
- `task`：
  - `default` — 直接翻译，只输出结果。
  - `terminology` — 在 `glossary` 中逐行写 `A 翻译成 B` / `A translates to B` 作为术语参考。
  - `style` — 用 `style` 约束译文风格。
  - `context` — 把 `background` 作为背景信息传入。
  - `structured` — 保持格式、缩进与占位符不变。
  - `custom` — 使用 `custom_prompt`，支持 `{text}` / `{source_text}` 与 `{target_lang}` 占位符。
- `max_tokens`：默认 1024。
- `strip_thinking`：去掉输出中的 `<think>…</think>` 片段与聊天标记。

提示词遵循[官方任务示例](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF#hy-mt3-translation-task-instruction-examples-chinese-english-comparison)，CLI 后端使用 GGUF 内置的 chat template；该模型没有默认 system prompt。

## 常见问题

- **`llama-cli not found`**：安装 llama.cpp，或设置 `HY_MT3_LLAMA_CLI`。
- **`No Hy-MT3 inference backend found`**：按上文安装任一后端。
- **下载失败**：检查网络/代理，设置 `HF_ENDPOINT`，或手动把准确文件名下载到 `models/hy_mt3/`。
- **显存不足**：改用更小量化（`IQ2`/`Q4_K_M`）、减小 `ctx_size`，或选择 `device=cpu`。
- **30B MoE 速度慢**：用 `Q4_K_M`、`device=gpu`；无 GPU 时可适当提高 `threads`。

<!-- 赞赏码：把图片放到 docs/wechat-pay.png，然后取消下面这行注释
![微信赞赏](docs/wechat-pay.png)
-->
