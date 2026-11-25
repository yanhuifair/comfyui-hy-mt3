"""GGUF download, backend discovery and text generation for Hy-MT3.

Two inference backends are supported:

1. ``llama-cli`` from llama.cpp (preferred: newest architectures such as hy_v3
   are supported as soon as llama.cpp supports them). Install with
   ``brew install llama.cpp`` or https://github.com/ggml-org/llama.cpp
2. ``llama-cpp-python`` (optional in-process fallback): ``pip install llama-cpp-python``

Set ``HY_MT3_BACKEND=cli|python`` to force one, or ``HY_MT3_LLAMA_CLI`` to
point at a non-PATH llama-cli executable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:  # package import (ComfyUI)
    from .hy_mt3_prompt import chat_prompt
    from .hy_mt3_registry import (
        PRESET_FILENAMES,
        download_url,
        spec_for_filename,
    )
except ImportError:  # direct module import (tests)
    from hy_mt3_prompt import chat_prompt
    from hy_mt3_registry import (
        PRESET_FILENAMES,
        download_url,
        spec_for_filename,
    )

try:  # ComfyUI is optional when this module is imported by the offline tests.
    import folder_paths
except ImportError:  # pragma: no cover - exercised only outside ComfyUI
    folder_paths = None

MODEL_FOLDER = "hy_mt3"
_ENGINE_CACHE: dict[tuple[Any, ...], Any] = {}


def model_folders() -> list[str]:
    """Directories searched for GGUF files, in priority order."""
    folders: list[str] = []
    if folder_paths is not None:
        get_paths = getattr(folder_paths, "get_folder_paths", None)
        if get_paths is not None:
            try:
                folders = list(get_paths(MODEL_FOLDER))
            except KeyError:  # folder type not registered yet
                folders = []
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir and os.path.join(models_dir, MODEL_FOLDER) not in folders:
            folders.append(os.path.join(models_dir, MODEL_FOLDER))
    if not folders:
        folders = [os.path.join(os.path.expanduser("~"), "ComfyUI", "models", MODEL_FOLDER)]
    return folders


def register_model_folder() -> None:
    if folder_paths is None:
        return
    add_path = getattr(folder_paths, "add_model_folder_path", None)
    if add_path is None:
        return
    for path in model_folders():
        add_path(MODEL_FOLDER, path)


def local_gguf_names() -> list[str]:
    names: list[str] = []
    for folder in model_folders():
        if not os.path.isdir(folder):
            continue
        for path in Path(folder).glob("*.gguf"):
            if path.name not in names:
                names.append(path.name)
    return names


def available_gguf_names() -> list[str]:
    names = list(PRESET_FILENAMES)
    for name in local_gguf_names():
        if name not in names:
            names.append(name)
    return names


@dataclass(frozen=True)
class HyMT3Model:
    """Serializable handle passed from the loader node to the translate node."""

    path: str
    name: str
    size: str
    quantization: str
    repo_id: str | None
    ctx_size: int
    ngl: int
    threads: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    seed: int
    timeout: int
    backend: str

    @property
    def label(self) -> str:
        return f"Hy-MT3 {self.size} {self.quantization}"


def resolve_model_path(name: str, override: str = "") -> str:
    if override.strip():
        return os.path.abspath(os.path.expanduser(override.strip()))
    if os.path.isabs(name):
        return os.path.abspath(os.path.expanduser(name))
    for folder in model_folders():
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(model_folders()[0], name)


def _download_with_urllib(url: str, destination: str, timeout: int = 60) -> None:
    """Stream a HF resolve URL into ``destination`` with resume support."""
    partial = destination + ".part"
    start = os.path.getsize(partial) if os.path.isfile(partial) else 0
    request = urllib.request.Request(url)
    if start:
        request.add_header("Range", f"bytes={start}-")

    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if start and status == 200:  # server ignored Range -> restart
            start = 0
        total = response.headers.get("Content-Length")
        total = int(total) + start if total else None
        mode = "ab" if start else "wb"
        done = start
        last_report = -1
        with open(partial, mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    percent = int(done * 100 / total)
                    if percent // 10 > last_report:
                        last_report = percent // 10
                        print(f"[Hy-MT3] download {percent}% ({done}/{total} bytes)")
    os.replace(partial, destination)


def ensure_model(path: str, name: str, auto_download: bool) -> None:
    if os.path.isfile(path):
        return
    if not auto_download:
        raise FileNotFoundError(
            f"GGUF not found: {path}\n"
            f"Put the file in {model_folders()[0]} or enable auto_download."
        )
    url = download_url(name)
    if url is None:
        raise FileNotFoundError(
            f"Unknown GGUF '{name}'. Pass an absolute path via model_path for custom files."
        )
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if endpoint:
        url = url.replace("https://huggingface.co", endpoint, 1)
    print(f"[Hy-MT3] downloading {name} from Hugging Face ...")
    try:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            _download_with_urllib(url, path)
        else:
            spec = spec_for_filename(name)
            downloaded = hf_hub_download(
                repo_id=spec.repo_id,
                filename=name,
                local_dir=os.path.dirname(path) or ".",
            )
            if os.path.abspath(downloaded) != os.path.abspath(path):
                os.replace(downloaded, path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {name}: {exc}\n"
            "Set HF_ENDPOINT if you use a mirror, or download the file manually into "
            f"{model_folders()[0]}."
        ) from exc
    print(f"[Hy-MT3] saved {path}")


def find_llama_cli() -> str | None:
    override = os.environ.get("HY_MT3_LLAMA_CLI", "").strip()
    if override:
        return override if os.path.isfile(override) else None
    return shutil.which("llama-cli")


def resolve_backend(preference: str = "auto") -> str:
    """Return 'cli' or 'python', raising a helpful error when neither works."""
    choice = (preference or "auto").lower()
    env_choice = os.environ.get("HY_MT3_BACKEND", "").strip().lower()
    if env_choice in ("cli", "python"):
        choice = env_choice

    if choice == "cli":
        if find_llama_cli() is None:
            raise RuntimeError(
                "llama-cli not found. Install llama.cpp (e.g. `brew install llama.cpp`) "
                "or set HY_MT3_LLAMA_CLI to the executable path."
            )
        return "cli"
    if choice == "python":
        try:
            import llama_cpp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run `pip install llama-cpp-python`."
            ) from exc
        return "python"

    if find_llama_cli() is not None:
        return "cli"
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "No Hy-MT3 inference backend found. Install llama.cpp "
            "(`brew install llama.cpp`) or `pip install llama-cpp-python`."
        )
    return "python"


def _cli_command(model: HyMT3Model, instruction: str, max_tokens: int) -> list[str]:
    cli = find_llama_cli()
    if cli is None:
        raise RuntimeError("llama-cli not found; check HY_MT3_LLAMA_CLI.")
    cmd = [
        cli,
        "-m", model.path,
        "-p", instruction,
        "--jinja",            # use the chat template embedded in the GGUF
        "-st",                # single turn, then exit
        "--no-escape",        # keep literal \n / quotes in the source text
        "-n", str(max_tokens),
        "-c", str(model.ctx_size),
        "--temp", str(model.temperature),
        "--top-p", str(model.top_p),
        "--top-k", str(0 if model.top_k < 0 else model.top_k),
        "--repeat-penalty", str(model.repeat_penalty),
        "-ngl", str(model.ngl),
    ]
    if model.threads > 0:
        cmd += ["-t", str(model.threads)]
    if model.seed >= 0:
        cmd += ["-s", str(model.seed)]
    return cmd


def _python_generate(model: HyMT3Model, instruction: str, max_tokens: int) -> str:
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("llama-cpp-python is not installed.") from exc

    key = (model.path, model.ctx_size, model.ngl, model.threads)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        engine = Llama(
            model_path=model.path,
            n_ctx=model.ctx_size,
            n_gpu_layers=-1 if model.ngl < 0 else model.ngl,
            n_threads=model.threads if model.threads > 0 else None,
            verbose=False,
        )
        _ENGINE_CACHE[key] = engine

    result = engine.create_completion(
        chat_prompt(instruction),
        max_tokens=max_tokens,
        temperature=model.temperature,
        top_p=model.top_p,
        top_k=max(model.top_k, 0),
        repeat_penalty=model.repeat_penalty,
        seed=model.seed if model.seed >= 0 else None,
    )
    return result["choices"][0]["text"]


def generate(
    model: HyMT3Model,
    instruction: str,
    max_tokens: int,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Run one translation and return the raw model output."""
    backend = resolve_backend(model.backend)
    if progress:
        progress(f"[Hy-MT3] backend={backend} model={model.path}")

    if backend == "python":
        return _python_generate(model, instruction, max_tokens)

    cmd = _cli_command(model, instruction, max_tokens)
    if progress:
        progress("[Hy-MT3] generating ...")
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=model.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"llama-cli timed out after {model.timeout}s. Increase the loader's timeout "
            "or lower max_tokens."
        ) from exc
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"llama-cli exited with {completed.returncode}:\n{tail}")

    output = completed.stdout
    # llama-cli may print a single interactive prompt marker on stdout.
    if output.startswith("> "):
        output = output[2:]
    return output.strip()
