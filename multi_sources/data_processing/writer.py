"""Implements the MultiSourceWriter class"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from lightning.pytorch.callbacks import BasePredictionWriter


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

    def __init__(self, root_dir, dt_max, dataset=None, multiple_realizations=False):
        """
        Args:
            root_dir (str or Path): The root directory where the predictions will be written.
            dt_max (pd.Timedelta): The maximum time delta in the dataset.
            dataset (MultiSourceDataset, optional): Dataset object that includes a method
                normalize(values, source_name, denorm=False) that can be used to denormalize the
                values before writing them to disk. If None, the values will
                not be denormalized.
            multiple_realizations (bool): If True, expects the predictions to include multiple
                realizations for each source. The values tensors are expected to have an additional
                leading dimension for the realizations.
        """
        super().__init__(write_interval="batch")
        self.root_dir = Path(root_dir)
        self.dt_max = dt_max
        self.dataset = dataset
        self.multiple_realizations = multiple_realizations

    def setup(self, trainer, pl_module, stage):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.targets_dir = self.root_dir / "targets"
        self.outputs_dir = self.root_dir / "outputs"
        self.targets_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        # Remove the info file if it exists
        info_file = self.root_dir / "info.csv"
        if info_file.exists():
            info_file.unlink()

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        pred, avail_tensors = prediction

        # We'll write to the info file in append mode
        info_file = self.root_dir / "info.csv"
        for source_index_pair, data in batch.items():
            source_name, index = source_index_pair  # Unpack the tuple
            with torch.no_grad():
                # DENORMALIZATION
                if self.dataset is not None:
                    device = data["values"].device
                    _, targets = self.dataset.normalize(
                        data["values"],
                        source_name,  # Use only source_name for normalization
                        denormalize=True,
                        leading_dims=1,
                        device=device,
                    )
                    if source_index_pair in pred:
                        _, pred[source_index_pair] = self.dataset.normalize(
                            pred[source_index_pair],
                            source_name,  # Use only source_name for normalization
                            denormalize=True,
                            leading_dims=1 + int(self.multiple_realizations),
                            device=device,
                        )

                # WRITING TARGETS
                # Create directory structure with source_name/index to organize by both source and index
                target_dir = self.targets_dir / source_name / str(index)
                target_dir.mkdir(parents=True, exist_ok=True)
                targets = targets.detach().cpu().float().numpy()
                # Append the lat/lon to the targets
                latlon = data["coords"].detach().cpu().float().numpy()
                dt = data["dt"].detach().cpu().float().numpy() * self.dt_max
                dt = pd.to_timedelta(dt).total_seconds()
                # The dimensions of the Datasets we'll create depend on the
                # dimensionality of the source.
                if len(targets.shape) == 4:  # 2d sources -> (B, C, H, W)
                    dims = ["samples", "H", "W"]
                elif len(targets.shape) == 2:  # 0d sources -> (B, C)
                    dims = ["samples"]
                # Fetch the names of the variables in the source
                source_obj = pl_module.sources[source_name]
                input_var_names = [v for v in source_obj.data_vars]
                # Create xarray Dataset for targets. Each channel in the targets
                # is a variable in the Dataset.
                coords = {
                    "samples": np.arange(targets.shape[0]),
                    "lat": (dims, latlon[:, 0]),
                    "lon": (dims, latlon[:, 1]),
                    "dt": (("samples"), dt),
                    "index": index,  # Add observation index as a coordinate
                }
                targets_ds = xr.Dataset(
                    {var: (dims, targets[:, i]) for i, var in enumerate(input_var_names)},
                    coords=coords,
                )
                targets_ds.to_netcdf(target_dir / f"{batch_idx}.nc")

                # WRITING PREDICTIONS
                if source_index_pair in pred:
                    # If no prediction was made for this source,
                    # skip it.
                    output_dir = self.outputs_dir / source_name / str(index)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Only keep variables that are in the source's output_vars
                    outputs = pred[source_index_pair].detach().cpu().float().numpy()
                    output_var_names = [v for v in source_obj.output_vars]
                    # Create xarray Dataset for outputs. Same principle as for targets.
                    # However, the predictions may have an additional dimension
                    # for multiple realizations.
                    if self.multiple_realizations:
                        dims = ["realization"] + dims
                        coords["realization"] = np.arange(outputs.shape[0])
                        outputs_ds = xr.Dataset(
                            {
                                var: (dims, outputs[:, :, i])  # Realization, batch, channel
                                for i, var in enumerate(output_var_names)
                            },
                            coords=coords,
                        )
                    else:
                        outputs_ds = xr.Dataset(
                            {
                                var: (dims, outputs[:, i])  # Batch, channel
                                for i, var in enumerate(output_var_names)
                            },
                            coords=coords,
                        )
                    outputs_ds.to_netcdf(output_dir / f"{batch_idx}.nc")

                batch_size = latlon.shape[0]
                # Convert the time deltas from fractions of dt_max to absolute durations
                # in hours.
                dt = data["dt"].detach().cpu().float().numpy() * self.dt_max
                dt = pd.Series(dt, name="dt")
                # Store the information about that source's data for that batch
                # in the info DataFrame.
                info_df = pd.DataFrame(
                    {
                        "source_name": [source_name] * batch_size,
                        "index": [index] * batch_size,  # Add observation index
                        "avail": avail_tensors[source_index_pair].detach().cpu().float().numpy(),
                        "batch_idx": [batch_idx] * batch_size,
                        "index_in_batch": np.arange(batch_size),
                        "dt": dt,
                        "channels": [targets.shape[1]] * batch_size,
                        "spatial_shape": [targets.shape[2:]] * batch_size,
                        "has_multiple_realizations": [self.multiple_realizations] * batch_size,
                    },
                )
                include_header = not info_file.exists()
            info_df.to_csv(info_file, mode="a", header=include_header, index=False)
