"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import torch
import torch.nn as nn

from multi_sources.losses.perceptual_loss import GeneralPerceptualLoss
from multi_sources.structure.base_reconstructor import MultisourceAbstractReconstructor

# Visualization imports
from multi_sources.utils.visualization import display_realizations


class MultisourceDeterministicReconstructor(MultisourceAbstractReconstructor):
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.
    The structure expects its input as a dict D {(source_name, index): map},
    where D[(source_name, index)] contains the following key-value pairs
    (all shapes excluding the batch dimension):
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

    The structure outputs a dict {(source_name, index): tensor} containing
    the predicted values for each source.
    """

    def __init__(
        self,
        sources,
        cfg,
        backbone,
        n_sources_to_mask,
        patch_size,
        values_dim,
        coords_dim,
        adamw_kwargs,
        lr_scheduler_kwargs,
        loss_max_distance_from_center=None,
        ignore_land_pixels_in_loss=False,
        normalize_coords_across_sources=False,
        mask_only_sources=None,
        forecasting_mode=False,
        perceptual_loss_weight=None,
        validation_dir=None,
        metrics={},
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            backbone (nn.Module): Backbone model to train.
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
            mask_only_sources (str or list of str): List of sources to mask. If None, all sources
                may be masked.
            forecasting_mode (bool): If True, will always mask all sources that are forecasted.
                A source is forecasted if its time delta is negative.
                Mutually exclusive with mask_only_sources.
            perceptual_loss_weight (float): Weight of the perceptual loss in the total loss.
                If None, no perceptual loss will be computed.
            validation_dir (optional, str or Path): Directory where to save the validation plots.
                If None, no plots will be saved.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true, masks, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
        """
        super().__init__(
            sources,
            cfg,
            backbone,
            n_sources_to_mask,
            patch_size,
            values_dim,
            coords_dim,
            adamw_kwargs,
            lr_scheduler_kwargs,
            loss_max_distance_from_center=loss_max_distance_from_center,
            ignore_land_pixels_in_loss=ignore_land_pixels_in_loss,
            normalize_coords_across_sources=normalize_coords_across_sources,
            mask_only_sources=mask_only_sources,
            forecasting_mode=forecasting_mode,
            validation_dir=validation_dir,
            metrics=metrics,
        )

        # [MASK] token that will replace the embeddings of the masked tokens
        self.mask_token = nn.Parameter(torch.randn(1, self.values_dim))

        # Optional perceptual loss
        if perceptual_loss_weight is not None:
            self.perceptual_loss_weight = perceptual_loss_weight
            self.perceptual_loss = GeneralPerceptualLoss()

    def embed(self, x):
        """Embeds the input sources. The embedded tensors' shapes depend on the dimensionality
        of the source:
        - For 2D sources: (B, h, w, Dv) for the values and (B, h, w, Dc) for the coordinates,
            where h = H // patch_size and w = W // patch_size.
        """
        # Embeds the values and coordinates using the embedding layers
        y = super().embed(x)
        # We know need to replace the masked values with a [MASK] token (specific to the
        # deterministic case).
        for source_index_pair, data in y.items():
            v = data["embedded_values"]
            where_masked = x[source_index_pair]["avail"] == 0
            where_masked = where_masked.view((where_masked.shape[0],) + (1,) * (v.dim() - 1))
            token = self.mask_token.view((1,) * (v.dim() - 1) + (-1,))
            data["embedded_values"] = torch.where(where_masked, token, v)

        return y

    def mask(self, x, masking_seed=None):
        """Masks a portion of the sources. A missing source cannot be chosen to be masked.
        Supposes that there are at least as many non-missing sources as the number of sources
        to mask. The number of sources to mask is determined by self.masking ratio.
        Masked sources have their values replaced by the [MASK] token.
        The availability flag is set to 0 where the source is masked.
        Args:
            x (dict of (source_name, index) to dict of str to tensor): The input sources.
            target_source (optional, str): If specified, the source to mask. If None, the source
                to mask is chosen randomly for each sample in the batch.
            masking_seed (int, optional): Seed for the random number generator used to select
                which sources to mask.
        Returns:
            masked_x (dict of (source_name, index) to dict of str to tensor):
                The input sources with a portion of the sources masked.
        """
        # Choose the sources to mask.
        avail_flags = super().select_sources_to_mask(x, masking_seed)
        # avail_flags[s][i] == 0 if the source s should be masked.

        # We just need to update the avail flag of each source. Where the flag is set to 0,
        # the values will be replaced by the [MASK] token in the embedding step.
        masked_x = {}
        for source_index_pair, data in x.items():
            # Copy the data to avoid modifying the original dict
            masked_data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
            avail_flag = avail_flags[source_index_pair]
            masked_data["avail"] = avail_flag
            # Set the availability mask to 0 everywhere for noised sources.
            # (!= from the avail flag, it's a mask of same shape
            masked_data["avail_mask"][avail_flag == 0] = 0
            masked_x[source_index_pair] = masked_data
        return masked_x

    def compute_loss(self, pred, batch, masked_batch):
        avail_flag = {
            source_index_pair: data["avail"] for source_index_pair, data in masked_batch.items()
        }
        targets = {
            source_index_pair: batch[source_index_pair]["values"] for source_index_pair in batch
        }

        # Only the keep the output variables from the ground truth
        targets = self.filter_output_variables(targets)
        # Compute the loss masks: a dict {(s,i): M} where M is a binary mask of shape
        # (B, ...) indicating which points should be considered in the loss.
        loss_masks = self.compute_loss_mask(batch, avail_flag)

        # Compute the MSE loss for each source
        losses = {}
        for source_index_pair in pred:
            # Compute the pointwise loss for each source.
            source_loss = (pred[source_index_pair] - targets[source_index_pair]).pow(2)
            # Multiply by the loss mask
            source_loss_mask = loss_masks[source_index_pair].unsqueeze(1).expand_as(source_loss)
            source_loss = source_loss * source_loss_mask
            # Compute the mean over the number of available points
            mask_sum = source_loss_mask.sum()
            if mask_sum == 0:
                # If all points are masked, we skip the loss computation for this source
                continue
            losses[source_index_pair] = source_loss.sum() / mask_sum

        # Optional perceptual loss
        if hasattr(self, "perceptual_loss"):
            for source_index_pair in pred:
                # Compute the perceptual loss over the masked samples
                masked = avail_flag[source_index_pair] == 0
                if not masked.any():
                    continue
                pred_masked = pred[source_index_pair][masked]
                target_masked = targets[source_index_pair][masked]
                mask_masked = batch[source_index_pair]["avail_mask"][masked].clone()
                lm_masked = batch[source_index_pair]["landmask"][masked]
                mask_masked[lm_masked == 1] = -1
                perceptual_loss = self.perceptual_loss(
                    pred_masked,
                    target_masked,
                    mask_masked,
                )
                losses[source_index_pair] += perceptual_loss * self.perceptual_loss_weight

        # Compute the total loss
        loss = sum(losses.values()) / len(losses)
        return loss

    def training_step(self, input_batch, batch_idx):
        batch_size = input_batch[list(input_batch.keys())[0]]["values"].shape[0]
        batch = self.preproc_input(input_batch)
        # Mask the sources
        masked_x = self.mask(batch)
        # Make predictions
        pred = self.forward(masked_x)
        # Compute the loss
        loss = self.compute_loss(pred, batch, masked_x)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, input_batch, batch_idx):
        batch = self.preproc_input(input_batch)
        # Mask the sources
        masked_batch = self.mask(batch)
        # Make predictions
        pred = self.forward(masked_batch)
        # Compute the loss
        loss = self.compute_loss(pred, batch, masked_batch)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        avail_flags = {
            source_index_pair: masked_batch[source_index_pair]["avail"]
            for source_index_pair in masked_batch
        }
        if self.validation_dir is not None and batch_idx % 50 == 0:
            # For every 30 batches, make a prediction and display it.
            if batch_idx % 50 == 0:
                display_realizations(
                    pred,
                    input_batch,
                    avail_flags,
                    self.validation_dir / f"realizations_{batch_idx}",
                    deterministic=True,
                    display_fraction=1.0,
                )

        # Evaluate the metrics
        y_true = {
            source_index_pair: batch[source_index_pair]["values"] for source_index_pair in batch
        }
        y_true = self.filter_output_variables(y_true)
        masks = self.compute_loss_mask(batch, avail_flags)
        for metric_name, metric in self.metrics.items():
            metric_res = metric(pred, y_true, masks)
            # Compute the average metric over all sources
            avg_res = torch.stack(list(metric_res.values())).mean()
            self.log(
                f"val_{metric_name}",
                avg_res,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        return loss

    def predict_step(self, input_batch, batch_idx):
        batch = self.preproc_input(input_batch)
        # Mask the sources
        masked_batch = self.mask(batch)
        # Make predictions
        pred = self.forward(masked_batch)
        # Fetch the availability flags for each source, so that
        # whatever processes the output can know which elements
        # were masked.
        avail_flags = {
            source_index_pair: masked_batch[source_index_pair]["avail"]
            for source_index_pair in masked_batch
        }
        return pred, avail_flags
