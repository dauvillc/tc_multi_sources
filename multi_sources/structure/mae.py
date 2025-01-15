"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import lightning.pytorch as pl
import torch
import torch.nn as nn
import numpy as np
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts
from multi_sources.models.utils import (
    normalize_coords_across_sources,
    embed_coords_to_sincos,
    remove_dots,
)
from multi_sources.models.embedding_layers import (
    CoordinatesEmbedding2d,
    SourcetypeEmbedding2d,
    CoordinatesEmbedding0d,
    SourceSpecificEmbedding0d,
)
from multi_sources.models.output_layers import SourcetypeProjection0d, SourcetypeProjection2d


class MultisourceMAE(pl.LightningModule):
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.
    The structure expects its input as a dict {source_name: map}, where each map contains the
    following key-value pairs (all shapes excluding the batch dimension):
    - "avail" is a scalar tensor of shape (B,) containing 1 if the element is available
        and -1 otherwise.
    - "values" is a tensor of shape (B, C, ...) containing the values of the source.
    - "landmask" is a tensor of shape (B, ...) containing the land-sea mask of the source.
    - "coords" is a tensor of shape (B, 2, ...) containing the spatial coordinates of the source.
    - "dt" is a scalar tensor of shape (B,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "dist_to_center" is a tensor of shape (B, ...) containing the distance
        to the center of the storm at each pixel, in km.
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
        coords_dim,  # Changed from coordinates_dim
        adamw_kwargs,
        lr_scheduler_kwargs,
        loss_max_distance_from_center,
        train_only_on_sources=[],
        exclude_sources_from_training=[],
        normalize_coords_across_sources=True,
        predict_dist_to_center=True,
        metrics={},
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            backbone (nn.Module): The backbone of the model.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            n_sources_to_mask (int): The number of sources to mask in each sample.
            patch_size (int): The size of the patches to split the images into.
            values_dim (int): The dimension of the values embeddings.
            coords_dim (int): The dimension of the coordinates embeddings.
            adamw_kwargs (dict): The arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): The arguments to pass to the learning rate scheduler.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            train_only_on_sources (list of str): If not empty, list of sources to train on
                exclusively. This means that at each training step, only those sources can
                be masked and then backpropagated on.
            exclude_sources_from_training (list of str): If not empty, list of sources to exclude
                from training. This means that at each training step, those sources will not be
                masked and the model will not backpropagate on them.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            predict_dist_to_center (bool): If True, adds the distance to the center of the storm
                as a target channel. The distance is not given as input to the backbone.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
        """
        super().__init__()
        self.sources = sources
        self.source_names = [source.name for source in sources]
        self.backbone = backbone
        self.n_sources_to_mask = n_sources_to_mask
        self.patch_size = patch_size
        self.values_dim = values_dim
        self.coords_dim = coords_dim  # Changed from self.coordinates_dim
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.adamw_kwargs = adamw_kwargs
        self.predict_dist_to_center = predict_dist_to_center
        self.metrics = metrics
        self.loss_max_distance_from_center = loss_max_distance_from_center
        self.train_only_on_sources = train_only_on_sources
        self.exclude_sources_from_training = exclude_sources_from_training
        self.normalize_coords_across_sources = normalize_coords_across_sources
        self.cfg = cfg
        self.save_hyperparameters(ignore=["backbone", "metrics"])

        if set(self.train_only_on_sources) & set(self.exclude_sources_from_training):
            raise ValueError(
                "The lists of sources to train on exclusively and to exclude from training "
                "should not have any element in common."
            )

        # Embedding and output projection layers
        # An embedding and an output layer for each source type
        # - We need to retrieve the list of each source type from the sources,
        #   as well as the number of context variables for each source type.
        self.sourcetypes_context_vars = {}
        self.sourcetype_embeddings = nn.ModuleDict()
        self.sourcetype_output_projs = nn.ModuleDict()
        self.sourcetype_coords_embeddings = nn.ModuleDict()
        self.source_to_type = {source.name: source.type for source in sources}
        for source in sources:
            # Only create the embedding layer for that source type if it doesn't exist yet
            if source.type not in self.sourcetypes_context_vars:
                self.sourcetypes_context_vars[source.type] = source.n_context_variables()
                n_output_channels = source.n_data_variables()
                n_output_channels += 1 if self.predict_dist_to_center else 0
                # Create the layers for that source type depending on
                # its dimensionality
                if source.dim == 2:
                    self.sourcetype_embeddings[source.type] = SourcetypeEmbedding2d(
                        source.n_data_variables(),
                        self.patch_size,
                        values_dim,
                    )
                    self.sourcetype_coords_embeddings[source.type] = CoordinatesEmbedding2d(
                        self.patch_size,
                        coords_dim,
                        source.n_context_variables(),
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection2d(
                        n_output_channels,
                        self.patch_size,
                        values_dim,
                    )
                elif source.dim == 0:
                    self.sourcetype_embeddings[source.type] = SourceSpecificEmbedding0d(
                        source.n_data_variables(),
                        values_dim,
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection0d(
                        n_output_channels,
                        values_dim,
                    )
                    self.sourcetype_coords_embeddings[source.type] = CoordinatesEmbedding0d(
                        coords_dim,
                    )
            else:
                # Check that the number of context variables is the same for all sources
                # of the same type
                if self.sourcetypes_context_vars[source.type] != source.n_context_variables():
                    raise ValueError(
                        f"Number of context variables is not "
                        "the same for all sources of type {source.type}"
                    )

        # learnable [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.values_dim))

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
            am = (~torch.isnan(v)[:, 0]).float()  # (B, H, W)
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
            # Potential context variables
            if "context" in data:
                ct = data["context"].float()
                ct = torch.nan_to_num(ct, nan=0)

            # Create two separate dictionaries: one for embedding input, one for loss computation
            embed_input = {
                "source_type": data["source_type"],
                "avail": data["avail"],
                "dt": dt,
                "coords": c,
                "values": v,
                "context": ct if "context" in data else None,
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
        """Embeds the input sources."""
        output = {}
        for source, data in x.items():
            # Embed the source's values
            source_type = self.source_to_type[source]
            v = self.sourcetype_embeddings[source_type](data)
            # Embed the source's coordinates
            c = self.sourcetype_coords_embeddings[source_type](data)
            # Save the layout of the tokens for the output projection. For example
            # a tokens_shape of (3, 2) means 3 tokens in the first dimension and 2 in the second.
            # That info is lost in the embedded sequences as they are flattened.
            tokens_shape = tuple(
                int(np.ceil(s / self.patch_size)) for s in data["values"].shape[-2:]
            )

            output[source] = {
                "tokens_shape": tokens_shape,
                "avail": data["avail"],
                "embedded_values": v,
                "embedded_coords": c,
            }
        return output

    def mask(self, x):
        """Masks a portion of the sources. A missing source cannot be chosen to be masked.
        Supposes that there are at least as many non-missing sources as the number of sources
        to mask. The number of sources to mask is determined by self.masking ratio.
        When a source is masked, all of its values are replaced by the [MASK] token.
        Tokens from missing sources are also replaced by the [MASK] token, but their avail flag
        is left to -1 so that they're not counted in the loss. For masked sources, the avail flag
        is set to 0.
        Args:
            x (dict of str to dict of str to tensor): The input sources, tokenized.
        Returns:
            masked_x (dict of str to dict of str to tensor): The input sources with a portion
                of the sources masked.
                The entry 'avail' is added, with its value being a tensor of shape (B, 1)
                containing 1 if the source is available, 0 if it was masked, and -1 if
                it was missing.
        """
        n_sources = len(x)
        any_elem = next(iter(x.values()))["embedded_values"]
        batch_size = any_elem.shape[0]
        device = any_elem.device

        # Select the sources to mask, which can differ between samples in the batch.
        # Missing sources cannot be masked.
        # Strategy: we'll generate a random noise tensor of shape (B, n_sources)
        # and for each row, mask the sources with the highest noise.
        noise = torch.rand((batch_size, n_sources), device=device)
        for i, (source, data) in enumerate(x.items()):
            # Multiply the noise by the availability mask (-1 for missing sources, 1 otherwise)
            noise[:, i] = noise[:, i] * data["avail"].squeeze(-1)
            # If the source is not included in the list of sources to train on, set its
            # noise to a value inferior to -1 so that it's not selected.
            if (self.train_only_on_sources and source not in self.train_only_on_sources) or (
                source in self.exclude_sources_from_training
            ):
                noise[:, i] = -2
        # Gather the indices of the sources to mask for each sample
        _, sources_to_mask = noise.topk(self.n_sources_to_mask, dim=1)  # (B, n_sources_to_mask)
        # Deduce a matrix M of shape (B, n_sources) such that M[b, i] = 1 if the source i
        # should be masked for the sample b, and 0 otherwise.
        masked_sources_matrix = torch.zeros(
            (batch_size, n_sources), dtype=torch.bool, device=device
        )  # (B, n_sources)
        masked_sources_matrix.scatter_(1, sources_to_mask, True)

        # Create the availability masks for the sources
        masked_x = {}
        for i, (source, data) in enumerate(x.items()):
            # Keep every entry in the source's dict in the output dict
            masked_x[source] = {k: v for k, v in data.items()}
            B, L, D = data["embedded_values"].shape
            whether_masked = masked_sources_matrix[:, i]  # (B,)
            # For samples in which the source is masked, set the availability to 0
            avail_tensor = torch.where(whether_masked, 0, data["avail"])
            masked_x[source]["avail"] = avail_tensor
            # For samples in which the source is masked or missing,
            # replace the tokens by the [MASK] token.
            values = data["embedded_values"]  # (B, L, D)
            masked_values = torch.where(
                (avail_tensor <= 0).view(B, 1, 1), self.mask_token.expand(B, L, D), values
            )
            masked_x[source]["embedded_values"] = masked_values
        return masked_x

    def forward(self, x):
        """Computes the forward pass of the model.
        Returns:
            pred (dict of str to tensor): The predicted values, as a dict
                {source_name: tensor} where the tensor has shape (B, C, ...).
            avail_tensors (dict of str to tensor): The availability tensors for each source,
                of shape (B,).
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
        x = self.mask(x)

        # Run the transformer backbone
        pred = self.backbone(x)

        # Output projections
        for source, v in pred.items():
            # Project from latent values space to output space using the output layer
            # corresponding to the source type
            pred[source] = self.sourcetype_output_projs[self.source_to_type[source]](
                v, tokens_shape=x[source]["tokens_shape"]
            )
        # For 2D sources, remove the padding
        for source, spatial_shape in spatial_shapes.items():
            pred[source] = pred[source][:, :, : spatial_shape[0], : spatial_shape[1]]

        # Compute availability tensors just before returning
        avail_tensors = {source: data["avail"] for source, data in x.items()}
        return pred, avail_tensors

    def step(self, batch, batch_idx, train_or_val):
        """Defines a training or validation step for the model."""
        batch = self.preproc_input(batch)
        batch_size = batch[list(batch.keys())[0]]["values"].shape[0]
        pred, avail_tensors = self.forward(batch)

        # Pass preprocessed batch to loss calculation
        losses = self.loss_fn(pred, batch, avail_tensors)
        # If len(losses) == 0, i.e. for all masked sources the tokens were missing,
        # raise an error.
        if len(losses) == 0:
            raise ValueError("No tokens to compute the loss on")
        # Compute the total loss
        loss = sum(losses.values()) / len(losses)
        self.log(
            f"{train_or_val}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
        )

        # Compute other metrics
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(pred, batch)
            self.log(f"{train_or_val}_{metric_name}", metric_value, batch_size=batch_size)
        return loss

    def loss_fn(self, y_pred, y_true, avail_tensors):
        """Computes the MSE between the predicted and true values. The loss is only computed
        on the masked tokens. If a max distance from the center is specified, the loss is also
        only computed for the tokens that are within this distance from the center.
        Args:
            y_pred (dict of str to tensor): The predicted values of shape
                (B, C, ...) for each source.
            y_true (dict of str to dict of str to tensor): The unmasked input data,
                with the following keys:
                - "values": The true values of shape (B, C, ...).
                - "avail_mask": The availability mask of shape (B, ...).
                - "dist_to_center": The distance to the center of the storm of shape (B, ...).
            avail_tensors (dict of str to tensor): The availability tensors for each source,
                of shape (B,).
        """
        losses = {}
        for source, true_data in y_true.items():
            B, C = true_data["values"].shape[:2]
            # If also predicting the distance to the center, concatenate it to the true values
            # after normalizing it. We can normalize by dividing by the max distance, which will
            # make the distance values between 0 and 1.
            if self.predict_dist_to_center:
                normalized_dist = true_data["dist_to_center"] / self.loss_max_distance_from_center
                true_data["values"] = torch.cat(
                    [true_data["values"], normalized_dist.unsqueeze(1)], dim=1
                )
            # We'll compute a mask M on the tokens of the source of shape (B, C, ...)
            # such that M[b, ...] = True if and only if the following conditions are met:
            # - The source was masked for the sample b (avail_tensors[b] == 0);
            # - the value at position ... was not missing (true_data["am"] == True);
            # - If self.loss_max_distance_from_center is not None, the token is within
            #   the specified distance from the center.
            where_avail = true_data["avail_mask"] >= 1
            when_masked = (avail_tensors[source] == 0).reshape(
                (B,) + (1,) * (where_avail.ndim - 1)
            )  # (B, 1, 1, ... one for each spatial dimension)
            mask = where_avail & when_masked  # (B, ... one for each spatial dimension)
            if self.loss_max_distance_from_center is not None:
                dist = true_data["dist_to_center"]
                mask = mask & (dist <= self.loss_max_distance_from_center)
            # Expand the mask to the number of channels in the source
            mask = mask.unsqueeze(1).expand_as(true_data["values"])  # (B, C, ...)
            # Compute the loss on the elements for which the loss maks is True
            true_values = true_data["values"][mask]
            pred_values = y_pred[source][mask]
            if true_values.numel() == 0:
                # If there are no tokens to compute the loss on, skip this source
                continue
            losses[source] = nn.functional.mse_loss(pred_values, true_values)
        return losses

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        """Defines a prediction step for the model.
        Returns:
            batch (dict of str to dict of str to tensor): The input batch.
            pred (dict of str to tensor): The predicted values.
            avail_tensors (dict of str to tensor): The availability tensors for each source.
        """
        # Save the non-normalized 'coords' entry of the batch, so that
        # we can save them with the predictions.
        original_coords = {source: x["coords"] for source, x in batch.items()}
        batch = self.preproc_input(batch)
        pred, avail_tensors = self.forward(batch)
        # Replace the 'coords' entry in the batch by the original coordinates
        for source, coords in original_coords.items():
            batch[source]["coords"] = coords
        return batch, pred, avail_tensors

    def configure_optimizers(self):
        """Configures the optimizer for the model.
        The optimizer is AdamW, with the default parameters.
        The learning rate goes through a linear warmup phase, then follows a cosine
        annealing schedule.
        """
        decay = self.adamw_kwargs.pop("weight_decay", 0.0)
        decay_params = {
            k: True for k, v in self.named_parameters() if "weight" in k and "norm" not in k
        }
        optimizer = torch.optim.AdamW(
            [
                {"params": [v for k, v in self.named_parameters() if k in decay_params]},
                {
                    "params": [v for k, v in self.named_parameters() if k not in decay_params],
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
