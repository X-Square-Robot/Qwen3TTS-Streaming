"""Backward-compatible re-export layer.

All path constants are now defined in ``tests.conftest``.
This module re-exports them for code that imports ``from tests.paths import ...``.
"""

from tests.conftest import (  # noqa: F401
    ENGINE_DIR,
    EXPORTED_DIR,
    MODELS_DIR,
    ONNX_DIR,
    REPO_ROOT,
    SHARED_TOKENIZER_DIR,
    TOKENIZER_DIR,
    VARIANT,
    VARIANT_MODEL_MAP,
    WEIGHTS_DIR,
)
