"""Implementation of custom PyTorch Lightning callbacks."""

import lightning.pytorch as pl


# Add a callback to check for unused parameters
class UnusedParameterChecker(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        unused = [n for n, p in pl_module.named_parameters() if p.requires_grad and p.grad is None]
        if unused:  # show them once per epoch
            rank = trainer.global_rank  # works with DDP too
            if rank == 0:
                print(
                    f"[Epoch {trainer.current_epoch}] " f"Parameters without gradients: {unused}"
                )
