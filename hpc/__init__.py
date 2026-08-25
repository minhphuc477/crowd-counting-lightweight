"""HPC-Lite: Hierarchical Probabilistic Crowd Counting for Robust Lightweight Deployment."""
import os

# Configure cache directories on F: disk (avoid C: drive)
_base_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache"))
os.environ.setdefault("HF_HOME", os.path.join(_base_cache, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_base_cache, "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

__version__ = "0.1.0"
