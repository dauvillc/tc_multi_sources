"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import lightning.pytorch as pl
import torch
import torch.nn as nn
import numpy as np
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts
from multi_sources.utils.image_processing import patches_to_img
from multi_sources.models.utils import (
    normalize_coords_across_sources,
    embed_coords_to_sincos,
    remove_dots,
)
from multi_sources.models.embedding_layers import SourceSpecificEmbedding2d, MasksEmbedding2d
from multi_sources.models.output_layers import SourceSpecificProjection2d


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
        masking_ratio,
        patch_size,
        values_dim,
        metadata_dim,
        adamw_kwargs,
        lr_scheduler_kwargs,
        loss_max_distance_from_center,
        use_attention_masks=False,
        normalize_coords_across_sources=True,
        metrics={},
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            backbone (nn.Module): The backbone of the model.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            masking_ratio (float): The ratio of tokens to mask. If mask_full_sources is True,
                ratio of sources to mask instead.
            patch_size (int): The size of the patches to split the images into.
            values_dim (int): The dimension of the values embeddings.
            metadata_dim (int): The dimension of the metadata embeddings.
            adamw_kwargs (dict): The arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): The arguments to pass to the learning rate scheduler.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            use_attention_masks (bool): If True, the model will use attention masks to
                ignore the tokens that are missing or masked.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
        """
        super().__init__()
        self.sources = sources
        self.source_names = [source.name for source in sources]
        self.backbone = backbone
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size
        self.values_dim = values_dim
        self.metadata_dim = metadata_dim
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.adamw_kwargs = adamw_kwargs
        self.metrics = metrics
        self.loss_max_distance_from_center = loss_max_distance_from_center
        self.use_attention_masks = use_attention_masks
        self.normalize_coords_across_sources = normalize_coords_across_sources
        self.save_hyperparameters(ignore=["backbone", "metrics"])

        # Embedding and output projection layers
        self.source_embeddings, self.output_projs = nn.ModuleDict(), nn.ModuleDict()
        for source in sources:
            # note: torch doesn't allow dots in keys of nn.ModuleDict, so we remove them
            self.source_embeddings[remove_dots(source.name)] = SourceSpecificEmbedding2d(
                source.n_data_variables(), source.shape, self.patch_size, values_dim, metadata_dim
            )
            self.output_projs[remove_dots(source.name)] = SourceSpecificProjection2d(
                source.n_data_variables(), source.shape, self.patch_size, values_dim
            )

        # Check if any source is 2D and create mask embedding layer if needed
        has_2d_source = any(len(source.shape) == 2 for source in sources)
        self.masks_embedding = None
        if has_2d_source:
            self.masks_embedding = MasksEmbedding2d(patch_size, values_dim)

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

            # Create two separate dictionaries: one for embedding input, one for loss computation
            embed_input = {
                "source_type": data["source_type"],
                "avail": data["avail"],
                "dt": dt,
                "coords": c,
                "values": v,
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
            # Check for NaN values in the tensors
            for k, v in input_[source].items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    if torch.isnan(v).any():
                        raise ValueError(f"NaN values found in source {source} tensor {k}")
        return input_

    def embed(self, x):
        """Embeds the input sources."""
        output = {}
        for source, data in x.items():
            # Get value and metadata embeddings
            v, m = self.source_embeddings[source](data)

            tokens_shape = tuple(
                int(np.ceil(s / self.patch_size)) for s in data["values"].shape[-2:]
            )

            output[source] = {
                "tokens_shape": tokens_shape,
                "avail": data["avail"],
                "embedded_values": v,
                "embedded_metadata": m,
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
        n_sources_to_mask = max(1, int(n_sources * self.masking_ratio))
        any_elem = next(iter(x.values()))["embedded_values"]
        batch_size = any_elem.shape[0]
        device = any_elem.device
        # Select the sources to mask, which can differ between samples in the batch.
        # Missing sources cannot be masked.
        noise = torch.rand((batch_size, n_sources), device=device)
        for i, (source, data) in enumerate(x.items()):
            # Multiply the noise by the availability mask (-1 for missing sources, 1 otherwise)
            noise[:, i] = noise[:, i] * data["avail"].squeeze(-1)
        _, sources_to_mask = noise.topk(n_sources_to_mask, dim=1)  # (B, n_sources_to_mask)
        masked_sources_matrix = torch.zeros(
            (batch_size, n_sources), dtype=torch.bool, device=device
        )  # (B, n_sources)
        masked_sources_matrix.scatter_(1, sources_to_mask, True)
        # Create the mask for the sources
        masked_x = {}
        for i, (source, data) in enumerate(x.items()):
            # First, retrieve each entry before masking
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

    def embed_after_mask(self, x, orig_data):
        """Computes the masks embeddings after masking has been applied.
        Args:
            x (dict): The masked and embedded data
            orig_data (dict): The original preprocessed data containing landmask and avail_mask
        Returns:
            y (dict): The masked and embedded data with the masks embeddings
        """
        if self.masks_embedding is None:
            return x

        for source, data in x.items():
            if len(orig_data[source]["values"].shape) == 4:  # Only for 2D sources
                # Create mask_data with only required fields from original data
                mask_data = {
                    "landmask": orig_data[source]["landmask"],
                    "avail_mask": orig_data[source]["avail_mask"].clone(),
                }
                # Only zero out availability mask for samples where avail <= 0
                mask_data["avail_mask"].masked_fill_(
                    data["avail"].view(-1, *([1] * (mask_data["avail_mask"].dim() - 1))) <= 0, 0
                )

                data["embedded_masks"] = self.masks_embedding(mask_data)

        return x

    def forward(self, x):
        """Computes the forward pass of the model.
        Returns:
            pred (dict of str to tensor): The predicted values, as a dict
                {source_name: tensor} where the tensor has shape (B, C, ...).
            avail_tensors (dict of str to tensor): The availability tensors for each source,
                of shape (B,).
        """
        preprocessed = x  # Store reference to preprocessed data
        x = self.embed(x)
        x = self.mask(x)
        # If a source is masked, its avail mask is zeroed out, so we can only
        # embed the availability mask after masking.
        x = self.embed_after_mask(x, preprocessed)

        attn_masks = None
        if self.use_attention_masks:
            attn_masks = {}
            for source, data in x.items():
                B, L = data["embedded_values"].shape[:2]
                attn_mask = data["avail"] <= 0  # (B,)
                attn_mask = attn_mask.view(B, 1).expand(B, L)
                attn_masks[source] = attn_mask

        pred = self.backbone(x, attn_masks)
        pred = {
            source: self.output_projs[source](v, x[source]["tokens_shape"])
            for source, v in pred.items()
        }
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
        for source, loss in losses.items():
            self.log(
                f"{train_or_val}_loss_{source}",
                loss,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        # Compute the total loss
        loss = sum(losses.values()) / len(losses)
        self.log(
            f"{train_or_val}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
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
            # We'll compute a mask M on the tokens of the source of shape (B, C, ...)
            # such that M[b, ...] = True if and only if the following conditions are met:
            # - The source was masked for the sample b (avail_tensors[b] == 0);
            # - the value at position ... was not missing (true_data["am"] == True);
            # - If self.loss_max_distance_from_center is not None, the token is within
            #   the specified distance from the center.
            mask = (true_data["avail_mask"] >= 1) & (avail_tensors[source].view(-1, 1, 1) == 0)
            if self.loss_max_distance_from_center is not None:
                dist = true_data["dist_to_center"]
                mask = mask & (dist <= self.loss_max_distance_from_center)
            # Expand the mask to the number of channels in the source
            mask = mask.unsqueeze(1).expand(-1, y_pred[source].shape[1], -1, -1)
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
