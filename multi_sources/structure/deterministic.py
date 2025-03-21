"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import torch
import torch.nn as nn

from multi_sources.structure.base_module import MultisourceAbstractReconstructor

# Visualization imports
from multi_sources.utils.visualization import display_realizations


class MultisourceDeterministicReconstructor(MultisourceAbstractReconstructor):
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
        """
        self.use_diffusion_t = False
        super().__init__(
            sources,
            backbone,
            cfg,
            n_sources_to_mask,
            patch_size,
            values_dim,
            coords_dim,
            adamw_kwargs,
            lr_scheduler_kwargs,
            loss_max_distance_from_center,
            ignore_land_pixels_in_loss,
            normalize_coords_across_sources,
            validation_dir,
            metrics,
        )

        # [MASK] token that will replace the embeddings of the masked tokens
        self.mask_token = nn.Parameter(torch.randn(1, self.values_dim))

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
        output = {}
        for source, data in y.items():
            v = data["embedded_values"]
            where_masked = x[source]["avail"] == 0
            v[where_masked] = self.mask_token.view((1,) * (v.dim() - 1) + (-1,))
            output[source] = {
                "embedded_values": v,
                "embedded_coords": data["embedded_coords"],
            }

        return output

    def mask(self, x, masking_seed=None):
        """Masks a portion of the sources. A missing source cannot be chosen to be masked.
        Supposes that there are at least as many non-missing sources as the number of sources
        to mask. The number of sources to mask is determined by self.masking ratio.
        Masked sources have their values replaced by the [MASK] token.
        The availability flag is set to 0 where the source is masked.
        Args:
            x (dict of str to dict of str to tensor): The input sources.
            target_source (optional, str): If specified, the source to mask. If None, the source
                to mask is chosen randomly for each sample in the batch.
            masking_seed (int, optional): Seed for the random number generator used to select
                which sources to mask.
        Returns:
            masked_x (dict of str to dict of str to tensor): The input sources with a portion
                of the sources masked.
        """
        # Choose the sources to mask.
        avail_flags = super().select_sources_to_mask(x, masking_seed)
        # avail_flags[s][i] == 0 if the source s should be masked.

        # We just need to update the avail flag of each source. Where the flag is set to 0,
        # the values will be replaced by the [MASK] token in the embedding step.
        masked_x = {}
        for source, data in x.items():
            # Copy the data to avoid modifying the original dict
            masked_data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
            masked_data["avail"] = avail_flags[source]
            masked_x[source] = masked_data
        return masked_x
    
    def compute_loss(self, pred, batch, masked_batch):
        # Retrieve the availability flag for each source updated after masking
        avail_flag = {source: data["avail"] for source, data in masked_batch.items()}

        # Filter the predictions and true values
        true_y = {source: batch[source]["values"] for source in batch}
        avail_mask = {source: batch[source]['avail_mask'] for source in batch}
        dist_to_center = {source: batch[source]['dist_to_center'] for source in batch}
        landmask = {source: batch[source]['landmask'] for source in batch}
        pred, true_y = super().apply_loss_mask(pred, true_y, avail_flag, avail_mask, dist_to_center, landmask)

        # Compute the loss
        losses = {}
        for source in pred:
            # Compute the loss for each source
            losses[source] = (pred[source] - true_y[source]).pow(2).mean()
        # If len(losses) == 0, i.e. for all masked sources the tokens were missing,
        # raise an error.
        if len(losses) == 0:
            raise ValueError("No tokens to compute the loss on")

        # Compute the total loss
        loss = sum(losses.values()) / len(losses)
        return loss

    def training_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]]["values"].shape[0]
        batch = self.preproc_input(batch)
        # Mask the sources
        masked_x = self.mask(batch)
        # Make predictions
        pred = self.forward(masked_x)
        # Compute the loss
        loss = self.compute_loss(pred, batch, masked_x)

        self.log(
            f"train_loss",
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
            f"val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        avail_flags = {source: masked_batch[source]["avail"] for source in masked_batch}
        if self.validation_dir is not None and batch_idx % 5 == 0:
            # For every 10 batches, make a prediction and display it.
            if batch_idx % 10 == 0:
                display_realizations(
                    pred,
                    input_batch,
                    avail_flags,
                    self.validation_dir / f"realizations_{batch_idx}",
                    deterministic=True,
                )

        # Evaluate the metrics
        for metric_name, metric in self.metrics.items():
            metric_res = metric(pred, batch, avail_flags)
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

    def predict_step(self, batch, batch_idx):
        # TODO
        pass
