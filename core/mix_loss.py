import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Union


class MixedLoss(nn.Module):
    """
    Primary loss always active; secondary loss kicks in
    *only* during training (module.train()) and after
    `start_epoch`.  Nothing extra is required in val/test.
    """

    def __init__(
        self,
        primary_loss: Union[nn.Module, Callable],
        secondary_loss: Optional[Union[nn.Module, Callable]] = None,
        alpha: float = 0.5,
        start_epoch: int = 0,
    ):
        super().__init__()
        self.primary_loss = primary_loss
        self.secondary_loss = secondary_loss
        self.alpha = alpha
        self.start_epoch = start_epoch
        self.current_epoch = 0  # call update_epoch() each epoch

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def update_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, y_pred, y_true) -> Dict[str, torch.Tensor]:
        p = self.primary_loss(y_pred, y_true)

        use_secondary = (
            self.secondary_loss is not None
            and self.training                       # train() vs eval()
            and self.current_epoch >= self.start_epoch
        )
        if use_secondary:
            s = self.secondary_loss(y_pred, y_true)
        else:
            s = torch.tensor(0.0, device=p.device, dtype=p.dtype)

        m = p + self.alpha * s
        return {"primary": p, "secondary": s, "mixed": m}