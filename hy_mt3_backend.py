"""Full-weight Hy-MT3 inference with Hugging Face Transformers.

The three official base repositories are loaded directly (BF16 safetensors),
following the code examples in their model cards. No GGUF or llama.cpp is
needed. Chat formatting is delegated to ``tokenizer.apply_chat_template`` so
the 7B template (different from 1.8B / 30B-A3B) is handled correctly.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:  # package import (ComfyUI)
    from .hy_mt3_registry import (
        MODEL_CHOICES,
        spec_for_repo,
        spec_for_size,
    )
except ImportError:  # direct module import (tests)
    from hy_mt3_registry import (
        MODEL_CHOICES,
        spec_for_repo,
        spec_for_size,
    )

try:  # ComfyUI is optional when this module is imported by the offline tests.
    import folder_paths
except ImportError:  # pragma: no cover - exercised only outside ComfyUI
    folder_paths = None

MODEL_FOLDER = "hy-mt3"
_CACHE: dict[tuple[Any, ...], tuple[Any, Any, str]] = {}
_CACHE_LOCK = threading.Lock()


class MissingDependency(RuntimeError):
    """Raised when transformers/torch is not available."""


def default_model_folder() -> str:
    if folder_paths is not None:
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return os.path.join(models_dir, MODEL_FOLDER)
    return os.path.join(os.path.expanduser("~"), "ComfyUI", "models", MODEL_FOLDER)


def register_model_folder() -> None:
    if folder_paths is None:
        return
    add_path = getattr(folder_paths, "add_model_folder_path", None)
    if add_path is not None:
        add_path(MODEL_FOLDER, default_model_folder())


def model_folders() -> list[str]:
    """Directories searched for locally stored full-weight models."""
    folders: list[str] = []
    if folder_paths is not None:
        get_paths = getattr(folder_paths, "get_folder_paths", None)
        if get_paths is not None:
            try:
                folders = list(get_paths(MODEL_FOLDER))
            except KeyError:
                pass
    folders.append(default_model_folder())
    for entry in os.environ.get("HY_MT3_MODEL_PATHS", "").split(os.pathsep):
        if entry.strip():
            folders.append(os.path.abspath(os.path.expanduser(entry.strip())))

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for snapshot in sorted(hub.glob("models--tencent--Hy-MT3-*/snapshots/*")):
        if snapshot.is_dir():
            folders.append(str(snapshot))

    unique: list[str] = []
    for folder in folders:
        if folder not in unique:
            unique.append(folder)
    return unique


def _size_from_config(path: Path) -> str | None:
    config_file = path / "config.json"
    if not config_file.is_file():
        return None
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    model_type = config.get("model_type")
    if model_type == "hy_v3":
        return "30B-A3B"
    if model_type == "hunyuan_v1_dense":
        return "7B" if config.get("hidden_size", 0) >= 4096 else "1.8B"
    for size in MODEL_CHOICES:
        if size.replace("-", "") in path.name.replace("-", ""):
            return size
    return None


def _local_size(source: str) -> str | None:
    """Infer the model size of a local directory from its config.json."""
    try:
        path = Path(source)
        if path.is_dir():
            return _size_from_config(path)
    except OSError:
        return None
    return None


def local_model_choices() -> list[str]:
    """Return labels like ``7B @ /path/to/local/model`` for local snapshots."""
    labels: list[str] = []
    candidates: list[Path] = []
    for folder in model_folders():
        base = Path(folder)
        if not base.is_dir():
            continue
        candidates.append(base)
        candidates.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    for candidate in candidates:
        size = _size_from_config(candidate)
        if size is None:
            continue
        label = f"{size} @ {candidate}"
        if label not in labels:
            labels.append(label)
    return labels


def available_models() -> list[str]:
    choices = list(MODEL_CHOICES)
    for label in local_model_choices():
        if label not in choices:
            choices.append(label)
    return choices


def resolve_source(choice: str, model_path: str = "") -> str:
    """Resolve the loader selection to a HF repo id or a local directory."""
    override = model_path.strip()
    if override:
        expanded = os.path.expanduser(override)
        if os.path.isdir(expanded):
            return os.path.abspath(expanded)
        if os.path.exists(expanded):
            raise ValueError(
                f"model_path must be a model directory, got a file: {override}"
            )
        # Not a local path: treat as a Hugging Face repo id (e.g. org/name).
        return override
    if " @ " in choice:
        return choice.split(" @ ", 1)[1]
    spec = spec_for_size(choice)
    if spec is None:
        raise ValueError(f"Unknown Hy-MT3 model: {choice}")
    return spec.repo_id


@dataclass
class HyMT3Model:
    """Handle passed from the loader node to the translate node."""

    source: str
    size: str
    tokenizer: Any
    model: Any
    device: str
    dtype: str
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    seed: int
    ctx_size: int

    @property
    def label(self) -> str:
        return f"Hy-MT3 {self.size} ({self.dtype}, {self.device})"


def _preferred_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    source: str,
    dtype_name: str = "bf16",
    device: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> HyMT3Model:
    """Load tokenizer + model, downloading from Hugging Face if necessary."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise MissingDependency(
            "PyTorch and transformers are required: "
            "pip install 'transformers>=5.6.0' (torch comes with ComfyUI)."
        ) from exc

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = dtype_map[dtype_name]

    spec = spec_for_repo(source) or spec_for_size(source)
    if spec is not None:
        size = spec.size
    else:
        # Local folder: infer the size from config.json so the sampling
        # defaults match the model family.
        size = _local_size(source) or "custom"

    with _CACHE_LOCK:
        cached = _CACHE.get((source, dtype_name, device))
    if cached is not None:
        tokenizer, model, actual_device = cached
        if progress:
            progress(f"[Hy-MT3] reuse cached {size} on {actual_device}")
    else:
        if progress:
            progress(f"[Hy-MT3] loading {source} ({dtype_name}) ...")
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
        try:
            import accelerate  # noqa: F401
        except ImportError:
            target = _preferred_device(device, torch)
            if progress:
                progress(f"[Hy-MT3] accelerate not installed; loading on {target}")
            model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
            model = model.to(target)
            actual_device = target
        else:
            kwargs["device_map"] = "auto" if device == "auto" else device
            model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
            actual_device = "auto" if device == "auto" else device
        model.eval()
        with _CACHE_LOCK:
            _CACHE[(source, dtype_name, device)] = (tokenizer, model, actual_device)

    # A fresh handle per call: the loader node may override sampling values,
    # and a cached instance would leak those overrides into later runs.
    sampling = spec or spec_for_size(size)
    return HyMT3Model(
        source=source,
        size=size,
        tokenizer=tokenizer,
        model=model,
        device=actual_device,
        dtype=dtype_name,
        temperature=sampling.temperature if sampling else 0.7,
        top_p=sampling.top_p if sampling else 0.6,
        top_k=sampling.top_k if sampling else 20,
        repeat_penalty=sampling.repeat_penalty if sampling else 1.05,
        seed=-1,
        ctx_size=8192,
    )


def generate(
    handle: HyMT3Model,
    instruction: str,
    max_tokens: int,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Translate one instruction using the model's own chat template."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency("PyTorch is required.") from exc

    if handle.seed >= 0:
        torch.manual_seed(handle.seed)

    messages = [{"role": "user", "content": instruction}]
    encoded = handle.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    device = getattr(handle.model, "device", None)
    if device is None:
        device = next(handle.model.parameters()).device
    if hasattr(encoded, "keys"):  # BatchEncoding
        inputs = {key: value.to(device) for key, value in encoded.items()}
        prompt_len = inputs["input_ids"].shape[-1]
    else:  # plain tensor
        input_ids = encoded.to(device)
        inputs = {"input_ids": input_ids}
        prompt_len = input_ids.shape[-1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "temperature": handle.temperature,
        "top_p": handle.top_p,
        "repetition_penalty": handle.repeat_penalty,
    }
    pad_token_id = handle.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = handle.tokenizer.eos_token_id
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id
    if handle.top_k > 0:
        gen_kwargs["top_k"] = handle.top_k
    if progress:
        progress(f"[Hy-MT3] generating on {device} ...")

    with torch.inference_mode():
        outputs = handle.model.generate(**inputs, **gen_kwargs)
    sequences = getattr(outputs, "sequences", outputs)
    if hasattr(sequences, "dim") and sequences.dim() == 1:
        sequences = sequences.unsqueeze(0)
    return handle.tokenizer.decode(
        sequences[0][prompt_len:], skip_special_tokens=True
    ).strip()
