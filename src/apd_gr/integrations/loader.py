from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager


@contextmanager
def integration_import_path(root):
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == root:
            sys.path.pop(0)


def load_integration_module(root, filename, stem):
    path = os.path.join(os.path.abspath(root), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required integration file not found: {path}")
    name = f"_unified_draft_{stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with integration_import_path(root):
        spec.loader.exec_module(module)
    return module


def load_diger_module(root, filename, stem):
    """Load DIGER while shielding its top-level `utils` import."""
    previous_utils = sys.modules.pop("utils", None)
    try:
        diger_utils = load_integration_module(root, "utils.py", "diger_utils")
        sys.modules["utils"] = diger_utils
        return load_integration_module(root, filename, stem)
    finally:
        sys.modules.pop("utils", None)
        if previous_utils is not None:
            sys.modules["utils"] = previous_utils
