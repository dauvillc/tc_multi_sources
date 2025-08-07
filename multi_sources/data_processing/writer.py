"""Implements the MultiSourceWriter class"""

import shutil
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
    - The predictions are of the form {source_name: v'} where v' are the predicted values.
        A source may not be included in the predictions.

    The writer automatically detects whether the predictions include multiple realizations
    (flow matching models) or a single prediction (deterministic models) based on the keys
    in the prediction dictionary.

    The predictions are written to disk in the following format:
    root_dir/targets/source_name/index/<sample_index>.nc
    root_dir/predictions/source_name/index/<sample_index>.nc
    root_dir/embeddings/source_name/index/<sample_index>.nc
    Additionally, the file root_dir/info.csv is written with the following columns:
    source_name, avail, dt, channels, spatial_shape, has_multiple_realizations.
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
        # Recreate the root directory if it already exists
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.targets_dir = self.root_dir / "targets"
        self.predictions_dir = self.root_dir / "predictions"
        self.targets_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        # Extract the components from the prediction dictionary
        if "sol" in prediction:
            # Flow matching model output format
            pred = prediction["sol"]
            avail_tensors = prediction["avail_flags"]
            # Flow matching models always have multiple realizations
            has_multiple_realizations = True
            # Extract predicted means if they exist
            pred_means = prediction.get("pred_mean", None)
            embeddings = None
        elif "predictions" in prediction:
            # Deterministic model output format
            pred = prediction["predictions"]
            avail_tensors = prediction["avail_flags"]
            # Deterministic models typically don't have multiple realizations
            has_multiple_realizations = False
            pred_means = None
            embeddings = prediction.get("embeddings", None)
        else:
            # Fallback for other formats
            raise ValueError(
                "Unexpected prediction format. Expected dictionary with 'sol'/'predictions' and 'avail_flags' keys."
            )
        # We'll create a "sample_index" coordinate that is unique across all batches
        sample_indexes = batch_indices

        # We'll write to the info file in append mode
        info_file = self.root_dir / "info.csv"
        for source_index_pair, data in batch.items():
            source_name, src_index = source_index_pair  # Unpack the tuple
            with torch.no_grad():
                # ================= DENORMALIZATION =================
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
                            leading_dims=1 + int(has_multiple_realizations),
                            device=device,
                        )
                    # Denormalize predicted means if they exist
                    if pred_means is not None and source_index_pair in pred_means:
                        _, pred_means[source_index_pair] = self.dataset.normalize(
                            pred_means[source_index_pair],
                            source_name,
                            denormalize=True,
                            leading_dims=1,  # Pred means have batch dimension but no realization dimension
                            device=device,
                        )

                # ================= TARGETS =================
                # Create directory structure with source_name/index to organize by both source and index
                target_dir = self.targets_dir / source_name / str(src_index)
                target_dir.mkdir(parents=True, exist_ok=True)
                targets = targets.detach().cpu().float().numpy()
                # Append the lat/lon to the targets
                latlon = data["coords"].detach().cpu().float().numpy()
                dt = data["dt"].detach().cpu().float().numpy() * self.dt_max
                dt = pd.to_timedelta(dt).total_seconds()
                # The dimensions of the Datasets we'll create depend on the
                # dimensionality of the source.
                if len(targets.shape) == 4:  # 2d sources -> (B, C, H, W)
                    dims = ["sample", "H", "W"]
                elif len(targets.shape) == 2:  # 0d sources -> (B, C)
                    dims = ["sample"]
                # Fetch the names of the variables in the source
                source_obj = pl_module.sources[source_name]
                input_var_names = [v for v in source_obj.data_vars]
                # Create xarray Dataset for targets. Each channel in the targets
                # is a variable in the Dataset.
                coords = {
                    "sample": sample_indexes,
                    "lat": (dims, latlon[:, 0]),
                    "lon": (dims, latlon[:, 1]),
                    "dt": (("sample"), dt),
                }
                targets_ds = xr.Dataset(
                    {var: (dims, targets[:, i]) for i, var in enumerate(input_var_names)},
                    coords=coords,
                )
                # Write each sample to a separate file. Ignore all samples for which the source
                # is unavailable (i.e., has an availability flag of -1).
                for k, sample_idx in enumerate(sample_indexes):
                    if avail_tensors[source_index_pair][k].item() == -1:
                        continue
                    sample_target_ds = targets_ds.sel(sample=sample_idx)
                    sample_target_ds.to_netcdf(target_dir / f"{sample_idx}.nc")

                # ================= PREDICTIONS =================
                if source_index_pair in pred:
                    # If no prediction was made for this source,
                    # skip it.
                    prediction_dir = self.predictions_dir / source_name / str(src_index)
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    # Only keep variables that are in the source's output_vars
                    predictions = pred[source_index_pair].detach().cpu().float().numpy()
                    output_var_names = [v for v in source_obj.output_vars]

                    # Create xarray Dataset for predictions. Same principle as for targets.
                    # However, the predictions may have an additional dimension
                    # for multiple realizations.
                    if has_multiple_realizations:
                        dims_with_realization = ["realization"] + dims
                        coords_with_realization = coords.copy()
                        coords_with_realization["realization"] = np.arange(predictions.shape[0])
                        predictions_ds = xr.Dataset(
                            {
                                var: (
                                    dims_with_realization,
                                    predictions[:, :, i],
                                )  # Realization, batch, channel
                                for i, var in enumerate(output_var_names)
                            },
                            coords=coords_with_realization,
                        )
                    else:
                        predictions_ds = xr.Dataset(
                            {
                                var: (dims, predictions[:, i])  # Batch, channel
                                for i, var in enumerate(output_var_names)
                            },
                            coords=coords,
                        )

                    # Write predicted means if available
                    if pred_means is not None and source_index_pair in pred_means:
                        pred_mean_outputs = (
                            pred_means[source_index_pair].detach().cpu().float().numpy()
                        )

                        # Create a dataset for the predicted means
                        pred_mean_ds = xr.Dataset(
                            {
                                f"pred_mean_{var}": (dims, pred_mean_outputs[:, i])
                                for i, var in enumerate(output_var_names)
                            },
                            coords=coords,
                        )

                        # Merge the datasets before writing to file
                        predictions_ds = xr.merge([predictions_ds, pred_mean_ds])

                    # Write each sample to a separate file
                    for k, sample_idx in enumerate(sample_indexes):
                        if avail_tensors[source_index_pair][k].item() == -1:
                            continue
                        sample_prediction_ds = predictions_ds.sel(sample=sample_idx)
                        sample_prediction_ds.to_netcdf(prediction_dir / f"{sample_idx}.nc")

                # ================= EMBEDDINGS =================
                if embeddings is not None and source_index_pair in embeddings:
                    # If embeddings are available, write them to disk
                    embedding_dir = self.root_dir / "embeddings" / source_name / str(src_index)
                    embedding_dir.mkdir(parents=True, exist_ok=True)
                    # There are three entries in the embeddings dict:
                    # - "coords": The coordinates of the source, of shape (B, ..., Dc)
                    #   where Dc is the coordinate embedding dimension.
                    # - "values": The values of the source, of shape (B, ..., Dv)
                    #   where Dv is the value embedding dimension.
                    # - "conditioning": The embedded conditioning, of shape (B, ..., Dg)
                    #   where Dg is the conditioning embedding dimension.
                    embedded_coords = (
                        embeddings[source_index_pair]["coords"].detach().cpu().float().numpy()
                    )
                    embedded_values = (
                        embeddings[source_index_pair]["values"].detach().cpu().float().numpy()
                    )
                    embedded_cond = (
                        embeddings[source_index_pair]["conditioning"]
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                    )
                    # Create a dataset for the embeddings
                    embed_dims = ["sample"]
                    if len(embedded_coords.shape) == 4:
                        embed_dims += ["lat", "lon"]
                    embedding_ds = xr.Dataset(
                        {
                            "coords": (embed_dims + ["coord_embed_dim"], embedded_coords),
                            "values": (embed_dims + ["value_embed_dim"], embedded_values),
                            "conditioning": (embed_dims + ["cond_embed_dim"], embedded_cond),
                        },
                        coords={
                            "sample": sample_indexes,
                        },
                    )
                    # Write each sample to a separate file
                    for k, sample_idx in enumerate(sample_indexes):
                        if avail_tensors[source_index_pair][k].item() == -1:
                            continue
                        sample_embedding_ds = embedding_ds.sel(sample=sample_idx)
                        sample_embedding_ds.to_netcdf(embedding_dir / f"{sample_idx}.nc")

                # ================= INFO FILE =================
                batch_size = latlon.shape[0]
                # Convert the time deltas from fractions of dt_max to absolute durations
                # in hours.
                dt = data["dt"].detach().cpu().float().numpy() * self.dt_max
                dt = pd.Series(dt, name="dt")

                # Add a flag to indicate whether predicted means are available
                has_pred_means = pred_means is not None and source_index_pair in pred_means

                # Store the information about that source's data for that batch
                # in the info DataFrame.
                info_df = pd.DataFrame(
                    {
                        "sample_index": sample_indexes,
                        "source_name": [source_name] * batch_size,
                        "source_index": [src_index] * batch_size,  # Add observation index
                        "avail": avail_tensors[source_index_pair].detach().cpu().float().numpy(),
                        "dt": dt,
                        "channels": [targets.shape[1]] * batch_size,
                        "spatial_shape": [targets.shape[2:]] * batch_size,
                        "has_multiple_realizations": [has_multiple_realizations] * batch_size,
                        "has_pred_means": [has_pred_means] * batch_size,  # New column
                    },
                )
                # Remove the rows where the source is not available
                info_df = info_df[info_df["avail"] != -1]
                include_header = not info_file.exists()
            info_df.to_csv(info_file, mode="a", header=include_header, index=False)
