"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import lightning.pytorch as pl
import torch
import torch.nn as nn
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts
from multi_sources.utils.image_processing import (
    img_to_patches,
    pad_to_next_multiple_of,
    pair,
)
from multi_sources.utils.image_processing import patches_to_img
from multi_sources.models.utils import normalize_coords_across_sources, remove_dots
from multi_sources.models.embedding_layers import (
    LinearEmbedding,
    SourceEmbedding,
    SharedSourceEmbedding,
)


class MultisourceMAE(pl.LightningModule):
    """Given a torch model which receives inputs from multiple sources, this module
    tokenizes the inputs, masks a portion of the tokens, and trains the model to reconstruct
    the masked tokens.
    The structure expects its input as a dict {source_name: map}, where each map contains the
    following key-value pairs (all shapes excluding the batch dimension):
    - "source_type" is a string containing the type of the source.
    - "avail" is a scalar tensor of shape (1,) containing 1 if the element is available
        and -1 otherwise.
    - "dt" is a scalar tensor of shape (1,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "context" is a tensor of shape (n_context_vars,) containing the context variables.
        Each data variable within a source has its own context variables, which are all
        concatenated into a single tensor to form CT.
    - "coords" is a tensor of shape (2, H, W) containing the latitude and longitude at each pixel.
    - "landmask" is a tensor of shape (H, W) containing the land mask.
    - "dist_to_center" is a tensor of shape (H, W) containing the distance
        to the center of the storm.
    - "values" is a tensor of shape (n_variables, H, W) containing the variables for the source.
    The structure uses an encoder and a decoder.
    Masked tokens are not included in the encoder's input nor output.
    The decoder receives the encoder's output as well as the masked tokens (whose value has been
    set to a learnable [MASK] token).
    """

    def __init__(
        self,
        source_names,
        encoder,
        decoder,
        cfg,
        masking_ratio,
        patch_size,
        values_dim,
        metadata_dim,
        adamw_kwargs,
        lr_scheduler_kwargs,
        loss_max_distance_from_center,
        share_source_embeddings=False,
        context_variables=None,
        output_convs=None,
        metrics={},
    ):
        """
        Args:
            source_names (list of str): Names of the sources.
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            masking_ratio (float): The ratio of tokens to mask.
            patch_size (tuple of int): The size of the patches to split the images into.
            values_dim (int): The dimension of the values embeddings.
            metadata_dim (int): The dimension of the metadata embeddings.
            adamw_kwargs (dict): The arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): The arguments to pass to the learning rate scheduler.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            share_source_embeddings (bool): If True, sources from the same type (e.g.
                "passive_microwave") will share the same embeddings, which will use
                context variables (e.g. the frequency of the sensor) as input.
                If False, each source will have its own embedding.
            n_context_variables (dict of str to list of str): Must be passed if
                share_source_embeddings is True. Maps each source type to the list of
                its context variables.
            output_convs (dict of str to nn.Module, optional): Map from source name to
                a convolutional layer to apply to the output of the model.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
        """
        super().__init__()
        self.source_names = source_names
        self.encoder = encoder
        self.decoder = decoder
        self.masking_ratio = masking_ratio
        self.patch_size = pair(patch_size)
        self.values_dim = values_dim
        self.metadata_dim = metadata_dim
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.adamw_kwargs = adamw_kwargs
        self.metrics = metrics
        self.loss_max_distance_from_center = loss_max_distance_from_center
        self.share_source_embeddings = share_source_embeddings
        self.save_hyperparameters(ignore=["encoder", "decoder", "metrics"])

        # Wether to use the ground truth for the unmasked tokens during prediction
        self.unmasked_tokens_use_groundtruth = False

        # Linear embeddings
        in_dim = self.patch_size[0] * self.patch_size[1]
        self.coord_embedding = LinearEmbedding(in_dim * 2, metadata_dim)  # latitude and longitude
        self.landmask_embedding = LinearEmbedding(in_dim, values_dim)
        self.value_embedding = LinearEmbedding(in_dim, values_dim)
        self.mask_embedding = LinearEmbedding(in_dim, values_dim)
        self.time_embedding = LinearEmbedding(1, metadata_dim)

        # The source embedding are learned, and will be summed both to
        # the values and coord embeddings
        if share_source_embeddings:
            if context_variables is None:
                raise ValueError(
                    "If share_source_embeddings is True, context_variables must be passed"
                )
            self.source_to_values_embedding = SharedSourceEmbedding(context_variables, values_dim)
            self.source_to_meta_embedding = SharedSourceEmbedding(context_variables, metadata_dim)
        else:
            self.source_to_values_embedding = SourceEmbedding(source_names, values_dim)
            self.source_to_meta_embedding = SourceEmbedding(source_names, metadata_dim)

        self.values_norm = nn.LayerNorm(values_dim)
        self.meta_norm = nn.LayerNorm(metadata_dim)

        # Projection of the predicted values to the original space
        self.output_proj = nn.Linear(self.values_dim, in_dim)

        self.output_convs = None
        if output_convs is not None:
            # Keys in nn.ModuleDict must not contain dots, so we replace them with _
            self.output_convs = nn.ModuleDict(remove_dots(output_convs))

        # learnable [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.values_dim))

    def preproc_input(self, x):
        # Normalize the coordinates across sources to make them relative instead of absolute
        # (i.e the min coord across all sources of a sample is always 0 and the max is 1).
        coords = [source["coords"] for source in x.values()]
        normed_coords = normalize_coords_across_sources(coords)

        input_ = {}
        for i, (source, data) in enumerate(x.items()):
            # Pad the image tensors to the next multiple of the patch size
            c = pad_to_next_multiple_of(normed_coords[i], self.patch_size, value=float("nan"))
            lm = pad_to_next_multiple_of(data["landmask"], self.patch_size, value=float("nan"))
            v = pad_to_next_multiple_of(data["values"], self.patch_size, value=float("nan"))
            d = pad_to_next_multiple_of(
                data["dist_to_center"], self.patch_size, value=float("nan")
            )
            # Don't modify the tensors in-place, as we need to keep the NaN values
            # for the loss computation
            dt = torch.nan_to_num(data["dt"], nan=-1.0)
            v = torch.nan_to_num(v, nan=0)
            lm = torch.nan_to_num(lm, nan=0)
            ct = torch.nan_to_num(data["context"], nan=0)
            # Where the coords are NaN, set them to -1, as the normalization set the non-nan values
            # to [0, 1]
            c = torch.nan_to_num(c, nan=-1)
            # For the distance tensor, fill the nan values with +inf
            d = torch.nan_to_num(d, nan=float("inf"))
            # Deduce the availability tensor from the distance tensor
            m = d != float("inf")
            input_[source] = {
                "source_type": data["source_type"],
                "avail": data["avail"],
                "dt": dt,
                "context": ct,
                "coords": c,
                "landmask": lm,
                "dist_to_center": d,
                "values": v,
                "mask": m,
            }
            # Check for NaN values in the tensors
            for k, v in input_[source].items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    if torch.isnan(v).any():
                        raise ValueError(f"NaN values found in source {source} tensor {k}")
        return input_

    def tokenize(self, x):
        """Converts the input sources into tokens."""
        output = {}
        for source, data in x.items():
            # c and v are image tensors of shape (b, c, h, w)
            c = img_to_patches(data["coords"], self.patch_size)
            v = img_to_patches(data["values"], self.patch_size)
            # lm, d and m are image tensors of shape (b, h, w)
            lm = img_to_patches(data["landmask"].unsqueeze(1), self.patch_size)
            d = img_to_patches(data["dist_to_center"].unsqueeze(1), self.patch_size)
            m = img_to_patches(data["mask"].unsqueeze(1), self.patch_size)
            output[source] = {
                "source_type": data["source_type"],
                "dt": data["dt"],
                "context": data["context"],
                "coords": c,
                "landmask": lm,
                "dist_to_center": d,
                "values": v,
                "mask": m,
            }
        return output

    def embed(self, x):
        """Embeds the input sources."""
        output = {}
        for source, data in x.items():
            c = data["coords"]
            B, L, _ = c.shape

            c = self.coord_embedding(c)
            lm = self.landmask_embedding(data["landmask"].float())
            v = self.value_embedding(data["values"].float())
            m = self.mask_embedding(data["mask"].float())

            dt = data["dt"].view(-1, 1, 1)
            embedded_dt = self.time_embedding(dt)

            # Source embeddings
            if self.share_source_embeddings:
                source_type = data["source_type"][0]  # The type is linked to the source
                # so it's the same for all samples from that source in the batch.
                source_to_values_embedding = self.source_to_values_embedding(
                    source_type, data["context"].unsqueeze(1)
                )
                source_to_meta_embedding = self.source_to_meta_embedding(
                    source_type, data["context"].unsqueeze(1)
                )
            else:
                source_to_values_embedding = self.source_to_values_embedding(source, (B, L))
                source_to_meta_embedding = self.source_to_meta_embedding(source, (B, L))

            # Sum the values, availability mask, land mask and source embeddings
            values = v + lm + m + source_to_values_embedding
            # Sum the coords, time and source embeddings to form the meta embeddings
            meta = c + embedded_dt + source_to_meta_embedding
            # Normalize the embeddings
            meta = self.meta_norm(meta)
            values = self.values_norm(values)

            output[source] = {
                "dt": dt,
                "embedded_dt": embedded_dt,
                "context": data["context"],
                "embedded_metadata": meta,
                "embedded_landmask": lm,
                "dist_to_center": data["dist_to_center"],
                "embedded_values": values,
                "mask": m,
            }
        return output

    def to_encoder_input(self, x):
        """Masks a portion of the tokens in the input sources.
        Args:
            x (dict of str to dict of str to tensor): The input sources, tokenized.
        Returns:
            encoder_x (dict of str to dict of str to tensor): The input for the encoder.
                encoder_x[source_name] is a map with the keys ['dt', 'embedded_dt',
                'embedded_source', 'embedded_metadata', 'embedded_values']
            token_perms (dict of str to tensor): The permutation of the full sequence used
                to shuffle the tokens.
            n_tokens (dict of str to int): The number of tokens in the original input.
            masked_tokens_indices (dict of str to tensor): The indices of the masked tokens, in
                a random order.
        """
        encoder_x = {}
        token_perms, n_tokens_map = {}, {}
        masked_tokens_indices = {}
        for source, data in x.items():
            m, v = data["embedded_metadata"], data["embedded_values"]
            B, L, D_v = v.shape  # batch size, sequence length, model dimension
            D_m = m.shape[-1]  # metadata dimension
            # Compute the number of tokens to keep based on the size of the input
            # and the masking ratio
            n_tokens = m.shape[1]
            n_kept = int(n_tokens * (1 - self.masking_ratio))
            # Randomly select the tokens to keep using the method from
            # "Masked Autoencoders Are Scalable Vision Learners" He et al. 2021
            # Shuffle the tokensnd keep the first n_masked.
            noise = torch.rand(B, L, device=m.device)
            perm = torch.argsort(noise, dim=1)
            kept_indices = perm[:, :n_kept].unsqueeze(-1)
            kept_indices_v = kept_indices.expand(-1, -1, D_v)
            kept_indices_m = kept_indices.expand(-1, -1, D_m)
            m = torch.gather(m, dim=1, index=kept_indices_m)
            v = torch.gather(v, dim=1, index=kept_indices_v)
            # Save the permutation and the original number of tokens for later
            # reconstruction of the original shape
            token_perms[source] = perm
            n_tokens_map[source] = n_tokens
            # Deduce the indices of the masked tokens
            masked_tokens_indices[source] = perm[:, n_kept:]

            encoder_x[source] = {
                "dt": data["dt"],
                "embedded_dt": data["embedded_dt"],
                "embedded_metadata": m,
                "embedded_values": v,
            }

        return encoder_x, token_perms, n_tokens_map, masked_tokens_indices

    def to_decoder_input(self, encoder_y, x, token_perms, n_tokens):
        """Concatenates the [MASK] token to the encoder output as many times as needed
        to obtain the original number of tokens.
        Args:
            encoder_y (dict of str to tensor): The output of the encoder.
            x (dict of str to dict of str to tensor): The input sources, tokenized.
            token_perms (dict of str to tensor): The permutation of the full sequence used
                to shuffle the tokens.
            n_tokens (dict of str to int): The number of tokens in the original input.
        """
        decoder_x = {}
        for source, x_data in x.items():
            y = encoder_y[source]
            B, L, D = y.shape
            # Get the permutation used to shuffle the tokens in the encoder input
            perm = token_perms[source]
            # Compute the number of tokens to add to retrieve the original shape
            n_tokens_to_add = n_tokens[source] - y.shape[1]
            # Concatenate the [MASK] token to the encoder output
            mask_tokens = self.mask_token.expand(B, n_tokens_to_add, -1)
            y = torch.cat((y, mask_tokens), dim=1)  # (b, n_tokens, d_model)
            # Unshuffle the tokens
            reverse_perm = torch.argsort(perm, dim=1).unsqueeze(-1).expand(-1, -1, D)
            y = torch.gather(y, dim=1, index=reverse_perm)

            decoder_x[source] = {
                "dt": x_data["dt"],
                "embedded_dt": x_data["embedded_dt"],
                "embedded_metadata": x_data["embedded_metadata"],
                "embedded_values": y,
            }
        return decoder_x

    def forward(self, x, padded_shapes):
        """Computes the forward pass of the model.
        Returns:
            pred (dict of str to tensor): The predicted values.
            masked_tokens_indices (dict of str to tensor): The indices of the masked tokens, in
                a random order.
            padded_shapes (dict of str to tuple of int): The shapes of the images after padding
                before tokenization.
        """
        x = self.embed(x)
        encoder_x, token_perms, n_tokens, masked_tokens_indices = self.to_encoder_input(x)
        encoder_y = self.encoder(encoder_x)

        decoder_x = self.to_decoder_input(encoder_y, x, token_perms, n_tokens)
        # Create the attention mask for the decoder: True where the tokens are masked,
        # and False elsewhere
        attn_masks = {}
        for source, indices in masked_tokens_indices.items():
            B, L = x[source]["embedded_values"].shape[:2]
            attn_mask = torch.zeros(B, L, device=indices.device, dtype=torch.bool)
            attn_mask.scatter_(1, indices, True)
            attn_masks[source] = attn_mask
        pred = self.decoder(decoder_x, attn_masks)
        # Project the predicted values to the original space
        pred = {source: self.output_proj(v) for source, v in pred.items()}
        # If an output convolutional layer is specified, apply it
        if self.output_convs is not None:
            # Convert the predictions back to images, apply the convolutional layer,
            # and convert them back to patches
            for source, v in pred.items():
                pH, pW = padded_shapes[source]
                v = patches_to_img(v, pH, pW, self.patch_size)
                v = self.output_convs[remove_dots(source)](v)
                v = img_to_patches(v, self.patch_size)
                pred[source] = v
        return pred, masked_tokens_indices

    def step(self, batch, batch_idx, train_or_val):
        """Defines a training or validation step for the model."""
        batch = self.preproc_input(batch)
        batch_size = batch[list(batch.keys())[0]]["values"].shape[0]
        # Save the shapes of the padded tensors to reconstruct the images
        padded_shapes = {source: x["values"].shape[-2:] for source, x in batch.items()}

        x = self.tokenize(batch)
        pred, masked_tokens_indices = self.forward(x, padded_shapes)

        # Compute the loss for each source
        losses = self.loss_fn(pred, x, masked_tokens_indices)
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
            metric_value = metric_fn(pred, x)
            self.log(f"{train_or_val}_{metric_name}", metric_value, batch_size=batch_size)
        return loss

    def loss_fn(self, y_pred, y_true, masked_token_indices):
        """Computes the MSE between the predicted and true values."""
        losses = {}
        for source, true_data in y_true.items():
            pred = y_pred[source]
            B, L, D = pred.shape
            # Retrieve the indices of the masked tokens
            masked_indices = masked_token_indices[source].unsqueeze(-1).expand(-1, -1, D)
            # For those tokens, retrieve the availability mask (as masked tokens can
            # correspond to missing data in the ground truth).
            mask = true_data["mask"].gather(1, masked_indices).float()
            # If a max distance from the center is specified, mask the tokens
            # that are too far from the center
            if self.loss_max_distance_from_center is not None:
                d_masked_tokens = true_data["dist_to_center"].gather(1, masked_indices)
                mask = mask * (d_masked_tokens < self.loss_max_distance_from_center).float()
            m_sum = mask.sum()
            if m_sum.item() == 0:
                # If all masked tokens are missing data, skip the loss computation
                # for this source
                continue
            # Retrieve the predicted and true values for the masked tokens
            pred = pred.gather(1, masked_indices)
            true_values = true_data["values"].gather(1, masked_indices)
            # Compute the MSE loss
            loss = nn.functional.mse_loss(pred, true_values, reduction="none")
            # Mask the loss where the ground truth is missing
            loss = (loss * mask).sum() / m_sum
            losses[source] = loss
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
        """
        batch = self.preproc_input(batch)
        # Save the shapes of the padded tensors to reconstruct the images
        padded_shapes = {source: x['values'].shape[-2:] for source, x in batch.items()}
        x = self.tokenize(batch)
        pred, masked_tokens_indices = self.forward(x, padded_shapes)
        output = {}
        for source, true_data in x.items():
            if self.unmasked_tokens_use_groundtruth:
                # If we want to use the ground truth for the unmasked tokens,
                # we'll replace the predicted values with the true values
                otp = true_data["values"].clone()
                B, L, D = pred[source].shape
                masked_indices = masked_tokens_indices[source].unsqueeze(-1).expand(-1, -1, D)
                otp.scatter_(1, masked_indices, pred[source].gather(1, masked_indices))
                output[source] = otp
            else:
                output[source] = pred[source].clone()
        # Convert the predictions back to images
        for source, v in output.items():
            pH, pW = padded_shapes[source]
            output[source] = patches_to_img(v, pH, pW, self.patch_size)
        return batch, output

    def use_groundtruth_for_unmasked_tokens(self, value):
        """Sets whether to use the ground truth for the unmasked tokens during prediction."""
        self.unmasked_tokens_use_groundtruth = value

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
