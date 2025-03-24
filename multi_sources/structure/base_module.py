"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

from abc import ABC, abstractmethod

import lightning.pytorch as pl
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Flow matching imports
from flow_matching.path import CondOTProbPath

# Local module imports
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts
from multi_sources.utils.solver import MultisourceEulerODESolver
from multi_sources.models.utils import (
    normalize_coords_across_sources,
    embed_coords_to_sincos,
)
from multi_sources.models.embedding_layers import (
    CoordinatesEmbedding2d,
    SourcetypeEmbedding2d,
    CoordinatesEmbedding0d,
    SourceSpecificEmbedding0d,
)
from multi_sources.models.output_layers import (
    SourcetypeProjection0d,
    SourcetypeProjection2d,
)

# Visualization imports
from multi_sources.utils.visualization import display_realizations


class MultisourceAbstractReconstructor(pl.LightningModule, ABC):
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.
    The structure expects its input as a dict D {source_name: map}, where D[source] contains the
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

    Besides the sources, D['id'] is a list of strings of length (B,) uniquely identifying the
    samples.

    The structure outputs a dict {source_name: tensor} containing the predicted values for each source.
    """

    def __init__(
        self,
        sources,
        backbone,
        cfg,
        n_sources_to_mask,
        patch_size,
        values_dim,
        coords_dim,
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
            backbone (nn.Module): Backbone of the model.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            n_sources_to_mask (int): Number of sources to mask in each sample.
            patch_size (int): Size of the patches to split the images into.
            values_dim (int): Dimension of the values embeddings.
            coords_dim (int): Dimension of the coordinates embeddings.
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
                A metric should have the signature metric(y_pred, batch, avail_flags, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
            **kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__()

        self.sources = {source.name: source for source in sources}
        self.source_names = [source.name for source in sources]
        self.backbone = backbone
        self.n_sources_to_mask = n_sources_to_mask
        self.patch_size = patch_size
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.validation_dir = validation_dir
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_max_distance_from_center = loss_max_distance_from_center
        self.ignore_land_pixels_in_loss = ignore_land_pixels_in_loss
        self.adamw_kwargs = adamw_kwargs
        self.metrics = metrics
        self.normalize_coords_across_sources = normalize_coords_across_sources

        # RNG that will be used to select the sources to mask
        self.source_select_gen = torch.Generator()

        # Initialize the embedding layers
        self.init_embedding_layers()

        # Save the configuration so that it can be loaded from the checkpoints
        self.cfg = cfg
        self.save_hyperparameters(ignore=["backbone", "metrics"])

    def init_embedding_layers(self):
        """Initializes the weights of the embedding layers."""
        # Embedding and output projection layers
        # An embedding and an output layer for each source type
        # - We need to retrieve the list of each source type from the sources,
        #   as well as the number of characs variables for each source type.
        self.sourcetypes_characs_vars = {}
        self.sourcetype_embeddings = nn.ModuleDict()
        self.sourcetype_output_projs = nn.ModuleDict()
        self.sourcetype_coords_embeddings = nn.ModuleDict()
        for source in self.sources.values():
            # Only create the embedding layer for that source type if it doesn't exist yet
            if source.type not in self.sourcetypes_characs_vars:
                self.sourcetypes_characs_vars[source.type] = source.n_charac_variables()
                n_output_channels = source.n_data_variables()
                # Create the layers for that source type depending on
                # its dimensionality
                if source.dim == 2:
                    self.sourcetype_embeddings[source.type] = SourcetypeEmbedding2d(
                        source.n_data_variables(),
                        self.patch_size,
                        self.values_dim,
                        use_diffusion_t=self.use_diffusion_t,
                    )
                    self.sourcetype_coords_embeddings[source.type] = CoordinatesEmbedding2d(
                        self.patch_size,
                        self.coords_dim,
                        source.n_charac_variables(),
                        use_diffusion_t=self.use_diffusion_t,
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection2d(
                        self.values_dim,
                        self.coords_dim,
                        n_output_channels,
                        self.patch_size,
                    )
                elif source.dim == 0:
                    self.sourcetype_embeddings[source.type] = SourceSpecificEmbedding0d(
                        source.n_data_variables(),
                        self.values_dim,
                    )
                    self.sourcetype_coords_embeddings[source.type] = CoordinatesEmbedding0d(
                        self.coords_dim, use_diffusion_t=self.use_diffusion_t
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection0d(
                        self.values_dim, self.coords_dim, n_output_channels
                    )
            else:
                # Check that the number of characs variables is the same for all sources
                # of the same type
                if self.sourcetypes_characs_vars[source.type] != source.n_charac_variables():
                    raise ValueError(
                        f"Number of characs variables is not "
                        "the same for all sources of type {source.type}"
                    )

    def preproc_input(self, x):
        # Normalize the coordinates across sources to make them relative instead of absolute
        # (i.e the min coord across all sources of a sample is always 0 and the max is 1).
        coords = [source["coords"] for source in x.values()]
        normed_coords = embed_coords_to_sincos(coords)
        if self.normalize_coords_across_sources:
            normed_coords = normalize_coords_across_sources(normed_coords)

        input_ = {}
        for i, (source, data) in enumerate(x.items()):
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

            input_[source] = {
                **embed_input,  # Data needed for embedding
                **loss_info,  # Additional data needed for loss computation
            }
        return input_

    def embed(self, x):
        """Embeds the input sources. The embedded tensors' shapes depend on the dimensionality
        of the source:
        - For 2D sources: (B, h, w, Dv) for the values and (B, h, w, Dc) for the coordinates,
            where h = H // patch_size and w = W // patch_size.
        """
        output = {}
        for source, data in x.items():
            # Embed the source's values
            source_type = self.sources[source].type
            v = self.sourcetype_embeddings[source_type](data)
            # Embed the source's coordinates
            c = self.sourcetype_coords_embeddings[source_type](data)

            output[source] = {
                "embedded_values": v,
                "embedded_coords": c,
            }

        return output

    def select_sources_to_mask(self, x, masking_seed=None):
        """Given a multi-sources batch, randomly selects a source to mask in each sample.
        Does not actually perform the masking.
        Args:
            x (dict of str to dict of str to tensor): The input sources.
            masking_seed (int, optional): Seed for the random number generator used to select
                which sources to mask.
        Returns:
            avail_flags (dict of str to tensor): The availability flags for each source,
                as tensors of shape (B,), such that:
                * avail_flags[s][i] == 1 if the source s is available for the sample i,
                * avail_flags[s][i] == 0 if the source s is masked for the sample i,
                * avail_flags[s][i] == -1 if the source s is missing for the sample i.
        """
        n_sources = len(x)
        any_elem = next(iter(x.values()))["values"]
        batch_size = any_elem.shape[0]
        device = any_elem.device

        if masking_seed is not None:
            self.source_select_gen.manual_seed(int(masking_seed))
        # Select the sources to mask, which can differ between samples in the batch.
        # Missing sources cannot be masked.
        # Strategy: we'll generate a random noise tensor of shape (B, n_sources)
        # and for each row, mask the sources with the highest noise.
        noise = torch.rand((batch_size, n_sources), generator=self.source_select_gen).to(device)
        for i, (source, data) in enumerate(x.items()):
            # Multiply the noise by the availability mask (-1 for missing sources, 1 otherwise)
            noise[:, i] = noise[:, i] * data["avail"].squeeze(-1)
        # Gather the indices of the sources to mask for each sample
        _, sources_to_mask = noise.topk(self.n_sources_to_mask, dim=1)  # (B, n_sources_to_mask)
        # Deduce a matrix M of shape (B, n_sources) such that M[b, i] = 1 if the source i
        # should be masked for the sample b, and 0 otherwise.
        masked_sources_matrix = torch.zeros(
            (batch_size, n_sources), dtype=torch.bool, device=device
        )  # (B, n_sources)
        masked_sources_matrix.scatter_(1, sources_to_mask, True)
        # Deduce the availability flags for each source
        avail_flags = {}
        for i, (source, data) in enumerate(x.items()):
            avail_flag = data["avail"].clone()
            avail_flag[masked_sources_matrix[:, i]] = 0
            avail_flags[source] = avail_flag
        return avail_flags

    @abstractmethod
    def mask(self, x, masking_seed=None):
        pass

    def forward(self, x):
        """Computes the forward pass of the model.
        Args:
            x (dict of str to dict of str to tensor): The input sources, masked.
        Returns:
            y (dict of str to tensor): The predicted values for each source.
        """
        # Save the shape of the tokens before they're embedded, so that we can
        # later remove the padding.
        spatial_shapes = {
            source: data["values"].shape[2:]
            for source, data in x.items()
            if len(data["values"].shape) > 2
        }
        # Embed and mask the sources
        x = self.embed(x)

        # Run the transformer backbone
        pred = self.backbone(x)

        for source, v in pred.items():
            # Embedded coords for the final modulation
            c = x[source]["embedded_coords"]
            # Project from latent values space to output space using the output layer
            # corresponding to the source type
            src_type = self.sources[source].type
            pred[source] = self.sourcetype_output_projs[src_type](v, c)
        # For 2D sources, remove the padding
        for source, spatial_shape in spatial_shapes.items():
            pred[source] = pred[source][..., : spatial_shape[0], : spatial_shape[1]]
        return pred

    def apply_loss_mask(
        self, y_pred, y_true, avail_flag, avail_mask, dist_to_center=None, landmask=None
    ):
        """Filters the predictions and groundtruth to to:
        - Only consider output-enabled variables.
        - Exclude the tokens that were missing in the ground truth.
        - Only consider the tokens that were masked in the input.
        - Optionally exclude the tokens that are too far from the center of the storm.
        - Optionally exclude the tokens that are on land.
        Args:
            y_pred (dict of str to tensor): The predictions, of shape (B, C, ...).
            y_true (dict of str to tensor): The groundtruth, of shape (B, C, ...).
            avail_flag (dict of str to tensor): The availability flag of each source, as tensor
                of shape (B,) containing 1 if the element is available, 0 if it was masked and
                -1 if it was missing.
            avail_mask (dict of str to tensor): The availability mask of each source, as tensor
                of shape (B, ...) containing 1 at spatial points where the element is available.
            dist_to_center (dict of str to tensor): The distance to the center of the storm
                for each source, as tensor of shape (B, ...).
            landmask (dict of str to tensor): The landmask for each source, as tensor of
                shape (B, ...).
        Returns:
            filtered_preds (dict of str to tensor): The filtered predictions, flattened.
            filtered_true_data (dict of str to tensor): The filtered groundtruth, flattened.
        """
        filtered_preds, filtered_gt = {}, {}
        for source, pred_s in y_pred.items():
            true_s = y_true[source]
            # Retrieve the mask of the output-enabled variables for the source,
            # and exclude them from the predictions and groundtruth.
            output_vars = self.sources[source].get_output_variables_mask()  # (C,)
            output_vars = torch.tensor(output_vars, device=pred_s.device)
            pred_s = pred_s[:, output_vars]
            true_s = true_s[:, output_vars]

            # We'll compute a mask M on the tokens of the source of shape (B, C, ...)
            # such that M[b, ...] = True if and only if the following conditions are met:
            # - The source was masked for the sample b (avail flag == 0);
            # - the value at position ... was not missing (true_data["am"] == True);
            loss_mask = avail_mask[source] >= 1  # (B, ...)
            loss_mask[avail_flag[source] != 0] = False
            # If a maximum distance from the center is specified, exclude the pixels
            # that are too far from the center from the loss computation.
            if self.loss_max_distance_from_center is not None:
                dist_mask = dist_to_center[source] <= self.loss_max_distance_from_center
                loss_mask = loss_mask & dist_mask
            # Optionally ignore the pixels that are on land
            if self.ignore_land_pixels_in_loss:
                loss_mask[landmask[source] > 0] = False
            # Apply the mask to the predictions and groundtruth
            loss_mask = loss_mask.unsqueeze(1).expand_as(pred_s)  # (B, C, ...)
            pred_s = pred_s[loss_mask]
            # If no elements are left, remove the source from the filtered predictions
            if pred_s.numel() == 0:
                continue
            filtered_preds[source] = pred_s
            filtered_gt[source] = true_s[loss_mask]
        return filtered_preds, filtered_gt

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
