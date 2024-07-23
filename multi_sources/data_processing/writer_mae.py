"""Implements the MultiSourceWriter class"""

import numpy as np
import pandas as pd
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter


class MultiSourceWriter(BasePredictionWriter):
    """Can be used as callback to a Lightning Trainer to write to disk the predictions of a model
    such that:
    - The targets are of the form {source_name: (s, dt, c, d, lm, v, m)}
        where v are the values to predict.
    - The outputs are of the form {source_name: v'} where v' are the predicted values.
        A source may not be included in the outputs.

    The predictions are written to disk in the following format:
    root_dir/targets/source_name/<batch_idx>.npy
    root_dir/outputs/source_name/<batch_idx>.npy
    Additionally, the file root_dir/info.csv is written with the following columns:
    source_name, batch_idx, dt
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
        batch, pred = prediction
        # We'll write to the info file in append mode
        info_file = self.root_dir / "info.csv"
        # An item in the batch is of the form {source_name: (s, dt, c, d, lm, v, m)}
        # We're interested in c (lat/lon) and v (values to predict)
        for source_name, (_, _, c, _, _, v, _) in batch.items():
            target_dir = self.targets_dir / source_name
            target_dir.mkdir(parents=True, exist_ok=True)
            targets = v.detach().cpu().numpy()
            # Append the lat/lon to the targets
            latlon = c.detach().cpu().numpy()
            targets_with_coords = np.concatenate([latlon, targets], axis=1)
            np.save(target_dir / f"{batch_idx}.npy", targets_with_coords)

            output_dir = self.outputs_dir / source_name
            output_dir.mkdir(parents=True, exist_ok=True)
            outputs = pred[source_name].detach().cpu().numpy()
            # The output will be saved
            np.save(output_dir / f"{batch_idx}.npy", outputs)

            s, dt = batch[source_name][:2]
            batch_size = latlon.shape[0]
            info_df = pd.DataFrame(
                {
                    "source_name": [source_name] * batch_size,
                    "batch_idx": batch_idx,
                    "dt": dt.detach().cpu().numpy(),
                },
            )
            include_header = not info_file.exists()
            info_df.to_csv(info_file, mode="a", header=include_header, index=False)
