"""APD-GR: adapter-based draft recommendation generation."""
from .modeling_parallel_draft import ParallelDraftModel
from .target_adapters import make_target_adapter

__all__ = ["ParallelDraftModel", "make_target_adapter"]
