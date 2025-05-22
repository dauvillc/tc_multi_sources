"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

from abc import ABC, abstractmethod

import lightning.pytorch as pl
import torch

from multi_sources.models.utils import (
    embed_coords_to_sincos,
    normalize_coords_across_sources,
)

# Local module imports
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts


class MultisourceAbstractModule(pl.LightningModule, ABC):
    """Given a torch model which receives inputs from multiple sources and outputs
    predictions for each source. This classes handles the preprocessing common to its
    subclasses and the filtering of the loss.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.
    The structure expects its input as a dict D {(source_name, index): map}, where D[(source_name, index)] contains the
    following key-value pairs (all shapes excluding the batch dimension):
    - "id" is a list of strings of length (B,) each uniquely identifying the elements.
    - "avail" is a scalar tensor of shape (B,) containing 1 if the element is available
        and -1 otherwise.
    - "values" is a tensor of shape (B, C, ...) containing the values of the source.
    - "landmask" is a tensor of shape (B, ...) containing the land-sea mask of the source.
    - "coords" is a tensor of shape (B, 2, ...) containing the spatial coordinates of the source.
    - "dt" is a scalar tensor of shape (B,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "dist_to_center" is a tensor of shape (B, ...) containing the distance
        to the center of the storm at each pixel, in km.

    The structure outputs a dict {(source_name, index): tensor} containing the predicted values
    for each source.
    """

    def __init__(
        self,
        sources,
        cfg,
        adamw_kwargs,
        lr_scheduler_kwargs,
        loss_max_distance_from_center=None,
        ignore_land_pixels_in_loss=False,
        normalize_coords_across_sources=False,
        validation_dir=None,
        metrics={},
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            adamw_kwargs (dict): Arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): Arguments to pass to the learning rate scheduler.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            ignore_land_pixels_in_loss (bool): If True, the pixels that are on land
                will be ignored in the loss computation.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            validation_dir (optional, str or Path): Directory where to save the validation plots.
                If None, no plots will be saved.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true, masks, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
            **kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__()

        self.sources = {source.name: source for source in sources}
        self.source_names = [source.name for source in sources]
        self.validation_dir = validation_dir
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_max_distance_from_center = loss_max_distance_from_center
        self.ignore_land_pixels_in_loss = ignore_land_pixels_in_loss
        self.adamw_kwargs = adamw_kwargs
        self.metrics = metrics
        self.normalize_coords_across_sources = normalize_coords_across_sources

        # Save the configuration so that it can be loaded from the checkpoints
        self.cfg = cfg
        self.save_hyperparameters(ignore=["backbone", "metrics"])

    def preproc_input(self, x):
        # Normalize the coordinates across sources to make them relative instead of absolute
        # (i.e the min coord across all sources of a sample is always 0 and the max is 1).
        coords = [source["coords"] for source in x.values()]
        normed_coords = embed_coords_to_sincos(coords)
        if self.normalize_coords_across_sources:
            normed_coords = normalize_coords_across_sources(normed_coords)

        input_ = {}
        for i, (source_index_pair, data) in enumerate(x.items()):
            c, v = normed_coords[i].float(), data["values"].float()
            lm, d = data["landmask"].float(), data["dist_to_center"].float()
            dt = data["dt"].float()
            # Deduce the availability mask level from where the values are missing
            # am = True where the values are available
            # /!\ The availability mask is different from the avail flag. The avail flag
            # is a single value for the whole source, which is -1 if the source is missing,
            # and 1 if it's available. The availability mask gives the availability of each
            # point in the source: 1 if the point is available, 0 if it's masked, -1 if missing.
            am = (~torch.isnan(v)[:, 0]).float() * 2 - 1  # (B, ...)
            # Don't modify the tensors in-place, as we need to keep the NaN values
            # for the loss computation
            dt = torch.nan_to_num(data["dt"], nan=-1.0)
            v = torch.nan_to_num(v, nan=0)
            lm = torch.nan_to_num(lm, nan=0)
            # Where the coords are NaN, set them to -1, as the normalization set the non-nan values
            # to [0, 1]
            c = torch.nan_to_num(c, nan=-1)
            # For the distance tensor, fill the nan values with +inf
            d = torch.nan_to_num(d, nan=float("inf"))
            # Potential characs variables
            if "characs" in data:
                ch = data["characs"].float()
                ch = torch.nan_to_num(ch, nan=0)

            # Create two separate dictionaries: one for embedding input, one for loss computation
            embed_input = {
                "avail": data["avail"],
                "dt": dt,
                "coords": c,
                "values": v,
                "characs": ch if "characs" in data else None,
            }

            loss_info = {
                "avail_mask": am,
                "landmask": lm,
                "dist_to_center": d,
            }

            input_[source_index_pair] = {
                **embed_input,  # Data needed for embedding
                **loss_info,  # Additional data needed for loss computation
            }
        return input_

    @abstractmethod
    def forward(self, x):
        pass

    def filter_output_variables(self, y_true):
        """Given the groundtruth, only keeps the output variables (as defined
        in the Source class).
        Args:
            y_true (dict of (source_name, index) to tensor): The groundtruth, of shape (B, C, ...).
        Returns:
            y_true_out (dict of (source_name, index) to tensor): The filtered groundtruth,
                of shape (B, C_filtered, ...). Doesn't modify the input dictionary.
        """
        y_true_out = {}
        for source_index_pair, true_s in y_true.items():
            # 1. Retrieve the list of output variables for the source and only keep those
            # in the groundtruth.
            source_name, _ = source_index_pair
            output_vars = self.sources[source_name].get_output_variables_mask()
            output_vars = torch.tensor(output_vars, device=true_s.device)
            true_s = true_s[:, output_vars]  # (B, C, ...) to (B, C_filtered, ...)
            y_true_out[source_index_pair] = true_s
        return y_true_out

    def compute_loss_mask(self, batch, avail_flag=None):
        """Computes a binary mask M on the tokens of the sources of shape (B, ...)
        such that M[b, ...] = True if and only if the following conditions are met:
        - Exclude the tokens that were missing in the ground truth.
        - Only consider the tokens that were masked in the input.
        - Optionally exclude the tokens that are too far from the center
          of the storm.
        - Optionally exclude the tokens that are on land.
        Args:
            batch (dict of (source_name, index) to dict of str to tensor): The input data, such that
                batch[(src, index)] includes the following keys:
                * avail_mask (dict of (source_name, index) to tensor): The availability mask of
                each source, as tensor of shape (B, ...) containing 1 at
                spatial points where the element is available.
                * dist_to_center (dict of (source_name, index) to tensor): The distance to the
                center of the storm for each source, as tensor of shape (B, ...).
                * landmask (dict of (source_name, index) to tensor): The landmask for each source,
                as tensor of shape (B, ...).
            avail_flag (dict of (source_name, index) to tensor): The availability flag of
               each source, as tensor of shape (B,) containing 1 if the
               element is available, 0 if it was masked and -1 if it was
               missing.
        Returns:
            masks (dict of (source_name, index) to tensor): The masks, of shape (B, ...),
                valued 1 where the token is available and 0 otherwise.
        """
        masks = {}
        for source_index_pair, source_data in batch.items():
            # We'll compute a mask M on the tokens of the source of shape (B, C, ...)
            # such that M[b, ...] = True if and only if the following conditions are met:
            # - The source was masked for the sample b (avail flag == 0);
            # - the value at position ... was not missing (true_data["am"] == True);
            loss_mask = source_data["avail_mask"] >= 1  # (B, ...)
            loss_mask[avail_flag[source_index_pair] != 0] = False
            # If a maximum distance from the center is specified, exclude the pixels
            # that are too far from the center from the loss computation.
            if self.loss_max_distance_from_center is not None:
                dist_mask = source_data["dist_to_center"] <= self.loss_max_distance_from_center
                loss_mask = loss_mask & dist_mask
            # Optionally ignore the pixels that are on land
            if self.ignore_land_pixels_in_loss:
                loss_mask[source_data["landmask"] > 0] = False
            masks[source_index_pair] = loss_mask

        return masks

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, input_batch, batch_idx):
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        """Configures the optimizer for the model.
        The optimizer is AdamW, with the default parameters.
        The learning rate goes through a linear warmup phase, then follows a cosine
        annealing schedule.
        """
        decay = self.adamw_kwargs.pop("weight_decay", 0.0)
        params = {k: v for k, v in self.named_parameters() if v.requires_grad}

        # Apply weight decay only to the weights that are not in the normalization layers
        decay_params = {k for k, _ in params.items() if "weight" in k and "norm" not in k}
        optimizer = torch.optim.AdamW(
            [
                # Parameters without decay
                {"params": [v for k, v in params.items() if k not in decay_params]},
                # Parameters with decay
                {
                    "params": [v for k, v in params.items() if k in decay_params],
                    "weight_decay": decay,
                },
            ],
            **self.adamw_kwargs,
        )

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            **self.lr_scheduler_kwargs,
        )
        return [optimizer], [scheduler]
