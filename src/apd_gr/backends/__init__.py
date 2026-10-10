"""Target-model execution backends shared by recommendation integrations."""

from .base import TargetBackend, select_context
from .causal import CausalBackend
from .t5 import T5Backend
from .t5_variants import CoFiRecBackend, DigerBackend, ReSIDBackend

__all__ = [
    "TargetBackend", "CausalBackend", "T5Backend", "CoFiRecBackend",
    "DigerBackend", "ReSIDBackend", "select_context",
]
