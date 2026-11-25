"""ComfyUI nodes for Tencent Hy-MT3 GGUF translation models.

Nodes:
  - Hy-MT3 Model Loader: pick/download a GGUF and set inference options.
  - Hy-MT3 Translate: translate text using a loaded model.
"""

from .hy_mt3_nodes import HyMT3ModelLoader, HyMT3Translate

NODE_CLASS_MAPPINGS = {
    "HyMT3ModelLoader": HyMT3ModelLoader,
    "HyMT3Translate": HyMT3Translate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HyMT3ModelLoader": "Hy-MT3 Model Loader",
    "HyMT3Translate": "Hy-MT3 Translate",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
