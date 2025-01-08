"""Implements the MultiSourceWriter class"""

import numpy as np
import pandas as pd
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter
import xarray as xr  # Import xarray


class MultiSourceWriter(BasePredictionWriter):
    """Can be used as callback to a Lightning Trainer to write to disk the predictions of a model
    such that:
    - The targets are of the form {source_name: D} where D is a dictionary with the following keys:
        - dt: The datetime of the observation
        - values: The values to predict as a tensor of shape (bs, C, ...)
            where ... are the spatial dimensions and C is the number of channels.
        - coords: The coordinates of each pixel as a tensor of shape (bs, 2, ...).
            The first channel is the latitude and the second channel is the longitude.
    - The outputs are of the form {source_name: v'} where v' are the predicted values.
        A source may not be included in the outputs.

    The predictions are written to disk in the following format:
    root_dir/targets/source_name/<batch_idx>.npy
    root_dir/outputs/source_name/<batch_idx>.npy
    Additionally, the file root_dir/info.csv is written with the following columns:
    source_name, avail, batch_idx, index_in_batch, dt, channels, spatial_shape.
    """

    def __init__(self, root_dir, dt_max, dataset=None):
        """
        Args:
            root_dir (str or Path): The root directory where the predictions will be written.
            dt_max (pd.Timedelta): The maximum time delta in the dataset.
            dataset (MultiSourceDataset, optional): Dataset object that includes a method
                normalize(values, source_name, denorm=False) that can be used to denormalize the
                values before writing them to disk. If None, the values will
                not be denormalized.
        """
        super().__init__(write_interval="batch")
        self.root_dir = Path(root_dir)
        self.dt_max = dt_max
        self.dataset = dataset

    def setup(self, trainer, pl_module, stage):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.targets_dir = self.root_dir / "targets"
        self.outputs_dir = self.root_dir / "outputs"
        self.targets_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        batch, pred, avail_tensors = prediction
        # We'll write to the info file in append mode
        info_file = self.root_dir / "info.csv"
        for source_name, data in batch.items():
            target_dir = self.targets_dir / source_name
            target_dir.mkdir(parents=True, exist_ok=True)
            targets = data['values'].detach().cpu().numpy()
            # Append the lat/lon to the targets
            latlon = data['coords'].detach().cpu().numpy()
            dt = data['dt'].detach().cpu().numpy() * self.dt_max
            # The dimensions of the Datasets we'll create depend on the
            # dimensionality of the source.
            if len(targets.shape) == 4:  # 2d sources -> (B, C, H, W)
                dims = ("samples", "channels", "H", "W")
                coord_dims = ("samples", "H", "W")
            elif len(targets.shape) == 2:  # 0d sources -> (B, C)
                dims = ("samples", "channels")
                coord_dims = ("samples",)
            # Create xarray Dataset for targets
            targets_ds = xr.Dataset(
                {
                    "targets": (dims, targets),
                },
                coords={
                    "samples": np.arange(targets.shape[0]),
                    "lat": (coord_dims, latlon[:, 0]),
                    "lon": (coord_dims, latlon[:, 1]),
                    "dt": (("samples"), dt),
                }
            )
            targets_ds.to_netcdf(target_dir / f"{batch_idx}.nc")

            if source_name in pred:
                # If no prediction was made for this source,
                # skip it.
                output_dir = self.outputs_dir / source_name
                output_dir.mkdir(parents=True, exist_ok=True)
                outputs = pred[source_name].detach().cpu().numpy()
                # Create xarray Dataset for outputs
                outputs_ds = xr.Dataset(
                    {
                        "outputs": (dims, outputs),
                    },
                    coords={
                        "samples": np.arange(outputs.shape[0]),
                        "lat": (coord_dims, latlon[:, 0]),
                        "lon": (coord_dims, latlon[:, 1]),
                        "dt": (("samples"), dt),
                    }
                )
                outputs_ds.to_netcdf(output_dir / f"{batch_idx}.nc")

            batch_size = latlon.shape[0]
            # We'll also save the time deltas of the samples. The deltas
            # in data['dt'] are coded as a fraction between 0 and 1 of the
            # maximum datetime in the dataset. We'll convert them into
            # actual Timedeltas here.
            dt = data['dt'].detach().cpu().numpy() * self.dt_max
            dt = pd.Series(dt, name="dt")
            info_df = pd.DataFrame(
                {
                    "source_name": [source_name] * batch_size,
                    "avail": avail_tensors[source_name].detach().cpu().numpy(),
                    "batch_idx": [batch_idx] * batch_size,
                    "index_in_batch": np.arange(batch_size),
                    "dt": dt,
                    "channels": [targets.shape[1]] * batch_size,
                    "spatial_shape": [targets.shape[2:]] * batch_size,
                },
            )
            include_header = not info_file.exists()
            info_df.to_csv(info_file, mode="a", header=include_header, index=False)
