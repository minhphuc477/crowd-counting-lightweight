"""Target supervision routing across terminal iterate y, initial carrier y0, or symmetric dual."""
from __future__ import annotations

from typing import Any
import torch


class TargetSupervisionRouter:
    """Routes target supervision across terminal measure y, initial carrier y0, or symmetric dual."""

    def __init__(self, target_mode: str = "y") -> None:
        if target_mode not in ("dual", "y0", "y"):
            raise ValueError(
                f"Unknown target supervision mode: '{target_mode}'. Expected 'dual', 'y0', or 'y'."
            )
        self.mode = target_mode

    def dispatch(
        self,
        fn: Any,
        y: torch.Tensor,
        y0: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Dispatch loss computation to y, y0, or their symmetric average."""
        if self.mode == "dual":
            l_y, l_y0 = fn(y, *args, **kwargs), fn(y0, *args, **kwargs)
            return 0.5 * (l_y + l_y0), {"y": l_y, "y0": l_y0}
        l_out = fn(y0 if self.mode == "y0" else y, *args, **kwargs)
        return l_out, {self.mode: l_out}
