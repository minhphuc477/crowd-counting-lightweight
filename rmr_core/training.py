from __future__ import annotations

"""Training Infrastructure Facade.

Re-exports training, checkpointing, reproducibility, and scheduling utilities:
  - rmr_core.checkpointing: build_checkpoint, safe_torch_save, safe_torch_load, register_safe_globals
  - rmr_core.provenance: get_git_info, compute_file_sha256
  - rmr_core.reproducibility: seed_everything, seed_worker, make_generator, save_rng_state, load_rng_state
  - rmr_core.schedulers: make_scheduler, make_cosine_scheduler, IdentityScheduler
"""

from .checkpointing import (
    build_checkpoint,
    register_safe_globals,
    safe_torch_load,
    safe_torch_save,
)
from .provenance import (
    compute_file_sha256,
    get_git_info,
)
from .reproducibility import (
    load_rng_state,
    make_generator,
    save_rng_state,
    seed_everything,
    seed_worker,
)
from .schedulers import (
    IdentityScheduler,
    make_cosine_scheduler,
    make_scheduler,
)

__all__ = [
    "IdentityScheduler",
    "build_checkpoint",
    "compute_file_sha256",
    "get_git_info",
    "load_rng_state",
    "make_cosine_scheduler",
    "make_generator",
    "make_scheduler",
    "register_safe_globals",
    "safe_torch_load",
    "safe_torch_save",
    "save_rng_state",
    "seed_everything",
    "seed_worker",
]
