"""Implements the MultiSourceWriter class"""

import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter


class MultiSourceWriter(BasePredictionWriter):
    """Can be used as callback to a Lightning Trainer to write to disk the predictions of a model
    such that:
    - The targets are of the form {source_name: S, DT, C, D, V} where V are the values to predict.
    - The outputs are of the form {source_name: V'} where V' are the predicted values.

    The predictions are written to disk in the following format:
    root_dir/targets/source_name/<batch_idx>.npy
    root_dir/outputs/source_name/<batch_idx>.npy
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (str or Path): The root directory where the predictions will be written.
        """
        super().__init__(write_interval="batch")
        self.root_dir = Path(root_dir)

    def setup(self, trainer, pl_module, stage):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.targets_dir = self.root_dir / "targets"
        self.outputs_dir = self.root_dir / "outputs"
        self.targets_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        for source_name, (_, _, _, _, v) in batch.items():
            source_target_dir = self.targets_dir / source_name
            source_output_dir = self.outputs_dir / source_name
            source_target_dir.mkdir(parents=True, exist_ok=True)
            source_output_dir.mkdir(parents=True, exist_ok=True)
            source_outputs = prediction[source_name].detach().cpu().numpy()
            source_targets = v.detach().cpu().numpy()
            np.save(source_target_dir / f"{batch_idx}.npy", source_targets)
            np.save(source_output_dir / f"{batch_idx}.npy", source_outputs)
