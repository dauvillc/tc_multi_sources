"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import lightning.pytorch as pl
import torch
import torch.nn as nn
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts
from multi_sources.utils.image_processing import img_to_patches, pad_to_next_multiple_of, pair
from multi_sources.utils.image_processing import patches_to_img
from multi_sources.models.utils import normalize_coords_across_sources


class MultisourceMAE(pl.LightningModule):
    """Given a torch model which receives inputs from multiple sources, this module
    tokenizes the inputs, masks a portion of the tokens, and trains the model to reconstruct
    the masked tokens.
    The structure receives inputs as a map {source_name: (A, S, DT, C, D, V)}, where:
    - A is a scalar tensor of shape (1,) containing 1 if the element is available and -1 otherwise.
    - S is a scalar tensor of shape (1,) containing the index of the source.
    - DT is a scalar tensor of shape (1,) containing the time delta between the synoptic time
      and the element's time, normalized by dt_max.
    - C is a tensor of shape (3, H, W) containing the latitude, longitude, and land-sea mask.
    - D is a tensor of shape (H, W) containing the distance to the center of the storm.
    - V is a tensor of shape (n_variables, H, W) containing the variables for the source.
    The structure uses an encoder and a decoder. The encoder receives the tokens as
    a list of tuples (S, DT, C, LM, V) and outputs a tensor of shape (b, n_tokens, d_model).
    Masked tokens are not included in the encoder's input nor output.
    The decoder receives the encoder's output as well as the masked tokens (whose value has been
    set to a learnable [MASK] token) and outputs a tensor of shape (b, n_tokens, n_variables).
    """

    def __init__(
        self,
        encoder,
        decoder,
        cfg,
        masking_ratio,
        patch_size,
        pixels_dim,
        coords_dim,
        lr_scheduler_kwargs,
        metrics={},
    ):
        """
        Args:
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            masking_ratio (float): The ratio of tokens to mask.
            patch_size (tuple of int): The size of the patches to split the images into.
            pixels_dim (int): The dimension of the pixel embeddings.
            coords_dim (int): The dimension of the coordinate embeddings.
            lr_scheduler_kwargs (dict): The arguments to pass to the learning rate scheduler.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.masking_ratio = masking_ratio
        self.patch_size = pair(patch_size)
        self.pixels_dim = pixels_dim
        self.coords_dim = coords_dim
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.metrics = metrics
        self.save_hyperparameters(ignore=["encoder", "decoder", "metrics"])

        # Linear embeddings for the patches (c, lm, v and m)
        in_dim = self.patch_size[0] * self.patch_size[1]
        self.coord_embedding = nn.Linear(in_dim * 2, coords_dim)  # latitude and longitude
        self.landmask_embedding = nn.Linear(in_dim, pixels_dim)
        self.value_embedding = nn.Linear(in_dim, pixels_dim)
        self.mask_embedding = nn.Linear(in_dim, pixels_dim)
        # Layernorm layers for the embeddings
        self.coord_layernorm = nn.LayerNorm(coords_dim)
        self.landmask_layernorm = nn.LayerNorm(pixels_dim)
        self.value_layernorm = nn.LayerNorm(pixels_dim)
        self.mask_layernorm = nn.LayerNorm(pixels_dim)

        # Projection of the predicted values to the original space
        self.output_proj = nn.Linear(self.pixels_dim, in_dim)

        # learnable [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.pixels_dim))

    def preproc_input(self, x):
        # x is a map {source_name: A, S, DT, C, D, V}
        # Normalize the coordinates across sources to make them relative instead of absolute
        # (i.e the min coord across all sources of a sample is always 0 and the max is 1).
        latlon = [x[3][:, :2] for x in x.values()]  # 3rd channel is the land mask
        normed_coords = normalize_coords_across_sources(latlon)
        input_ = {}
        for i, (source, (a, s, dt, c, d, v)) in enumerate(x.items()):
            # C[:, 2:3] is the land mask, which we'll split from the latitude and longitude
            lm = c[:, 2:3]
            # Pad the image tensors to the next multiple of the patch size
            c = pad_to_next_multiple_of(normed_coords[i], self.patch_size, value=float("nan"))
            lm = pad_to_next_multiple_of(lm, self.patch_size, value=float("nan"))
            v = pad_to_next_multiple_of(v, self.patch_size, value=float("nan"))
            d = pad_to_next_multiple_of(d, self.patch_size, value=float("nan"))
            # Don't modify the tensors in-place, as we need to keep the NaN values
            # for the loss computation
            dt = torch.nan_to_num(dt, nan=-1.0)
            v = torch.nan_to_num(v, nan=0)
            # Where the coords are NaN, set them to -1, as the normalization set the non-nan values
            # to [0, 1]
            c = torch.nan_to_num(c, nan=-1)
            # Where the land mask is NaN, set it to 0
            lm = torch.nan_to_num(lm, nan=0)
            # For the distance tensor, fill the nan values with +inf
            d = torch.nan_to_num(d, nan=float("inf"))
            # Deduce the availability tensor from the distance tensor
            m = d != float("inf")
            input_[source] = (s, dt, c, d, lm, v, m)
            # Check for NaN values in the tensors
            for k, tensor in enumerate(input_[source]):
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN values found in source {source} tensor {k}")
        return input_

    def tokenize(self, x):
        """Converts the input sources into tokens."""
        output = {}
        for source, (s, dt, c, d, lm, v, m) in x.items():
            # a, s and dt are scalars, so don't need to be tokenized
            # c, lm and v are image tensors of shape (b, c, h, w)
            c = img_to_patches(c, self.patch_size)
            v = img_to_patches(v, self.patch_size)
            lm = img_to_patches(lm, self.patch_size)
            m = img_to_patches(m.unsqueeze(1), self.patch_size)
            # d doesn't need to be tokenized, as it won't be fed to the model.
            # It's only used in the loss computation, which is done on the original shape.
            output[source] = (s, dt, c, d, lm, v, m)
        return output

    def embed(self, x):
        """Embeds the input sources."""
        output = {}
        for source, (s, dt, c, d, lm, v, m) in x.items():
            c = self.coord_layernorm(self.coord_embedding(c))
            lm = self.landmask_layernorm(self.landmask_embedding(lm))
            v = self.value_layernorm(self.value_embedding(v))
            m = self.mask_layernorm(self.mask_embedding(m.float()))
            output[source] = (s, dt, c, d, lm, v, m)
        return output

    def to_encoder_input(self, x):
        """Masks a portion of the tokens in the input sources.
        Args:
            x (dict of str to tuple of tensors): The input sources, tokenized.
        Returns:
            encoder_x (list of tuple of tensors): Fraction of the input tokens,
                which can be fed to the encoder.
            token_perms (dict of str to tensor): The permutation of the full sequence used
                to shuffle the tokens.
            n_tokens (dict of str to int): The number of tokens in the original input.
            masked_tokens_indices (dict of str to tensor): The indices of the masked tokens, in
                a random order.
        """
        encoder_x = []
        token_perms, n_tokens_map = {}, {}
        masked_tokens_indices = {}
        for source, (s, dt, c, d, lm, v, m) in x.items():
            B, L, D = v.shape  # batch size, sequence length, model dimension
            # Compute the number of tokens to keep based on the size of the input
            # and the masking ratio
            n_tokens = c.shape[1]
            n_kept = int(n_tokens * (1 - self.masking_ratio))
            # Randomly select the tokens to keep using the method from
            # "Masked Autoencoders Are Scalable Vision Learners" He et al. 2021
            # Shuffle the tokensnd keep the first n_masked.
            noise = torch.rand(B, L, device=c.device)
            perm = torch.argsort(noise, dim=1)
            kept_indices = perm[:, :n_kept].unsqueeze(-1).expand(-1, -1, D)
            c = torch.gather(c, dim=1, index=kept_indices)
            lm = torch.gather(lm, dim=1, index=kept_indices)
            v = torch.gather(v, dim=1, index=kept_indices)
            m = torch.gather(m, dim=1, index=kept_indices)
            # Save the permutation and the original number of tokens for later
            # reconstruction of the original shape
            token_perms[source] = perm
            n_tokens_map[source] = n_tokens
            encoder_x.append((s, dt, c, lm, v, m))
            # Deduce the indices of the masked tokens
            masked_tokens_indices[source] = perm[:, n_kept:]

        return encoder_x, token_perms, n_tokens_map, masked_tokens_indices

    def to_decoder_input(self, encoder_y, x, token_perms, n_tokens):
        """Concatenates the [MASK] token to the encoder output as many times as needed
        to obtain the original number of tokens.
        Args:
            encoder_y (list of tensors): The output of the encoder.
            x (list of tuple of tensors): The input sources, tokenized.
            token_perms (dict of str to tensor): The permutation of the full sequence used
                to shuffle the tokens.
            n_tokens (dict of str to int): The number of tokens in the original input.
        """
        decoder_x = []
        for y, (source, (s, dt, c, d, lm, _, m)) in zip(encoder_y, x.items()):
            B, L, D = y.shape
            # Get the permutation used to shuffle the tokens in the encoder input
            perm = token_perms[source]
            # Compute the number of tokens to add to retrieve the original shape
            n_tokens_to_add = n_tokens[source] - y.shape[1]
            # Concatenate the [MASK] token to the encoder output
            mask_tokens = self.mask_token.expand(c.shape[0], n_tokens_to_add, -1)
            y = torch.cat((y, mask_tokens), dim=1)  # (b, n_tokens, d_model)
            # Unshuffle the tokens
            reverse_perm = torch.argsort(perm, dim=1).unsqueeze(-1).expand(-1, -1, D)
            y = torch.gather(y, dim=1, index=reverse_perm)
            # Since the input to the decoder is not the initial pixels, but the tokens
            # processed by the encoder, the availability tensor is not needed.
            m = torch.zeros_like(m)
            decoder_x.append((s, dt, c, lm, y, m.float()))
        return decoder_x

    def forward(self, x):
        """Computes the forward pass of the model.
        Returns:
            pred (dict of str to tensor): The predicted values.
            masked_tokens_indices (dict of str to tensor): The indices of the masked tokens, in
                a random order.
        """
        x = self.embed(x)
        encoder_x, token_perms, n_tokens, masked_tokens_indices = self.to_encoder_input(x)
        encoder_y = self.encoder(encoder_x)
        decoder_x = self.to_decoder_input(encoder_y, x, token_perms, n_tokens)
        pred = self.decoder(decoder_x)
        # Project the predicted values to the original space
        pred = [self.output_proj(p) for p in pred]
        # Rebuild a map {source_name: pred} from the list of predictions
        pred = {source: v for source, v in zip(x.keys(), pred)}
        return pred, masked_tokens_indices

    def step(self, batch, batch_idx, train_or_val):
        """Defines a training or validation step for the model."""
        batch = self.preproc_input(batch)
        x = self.tokenize(batch)
        pred, masked_tokens_indices = self.forward(x)
        # Compute and log the loss
        loss = self.loss_fn(pred, x, masked_tokens_indices)

        self.log(f"{train_or_val}_loss", loss, prog_bar=True, on_epoch=True)
        # Compute metrics
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(pred, x)
            self.log(f"{train_or_val}_{metric_name}", metric_value)
        return loss

    def loss_fn(self, y_pred, y_true, masked_token_indices):
        """Computes the MSE between the predicted and true values."""
        losses = []
        for source, (_, _, _, _, _, v, m) in y_true.items():
            pred = y_pred[source]
            B, L, D = pred.shape
            # Compute the loss only for the masked tokens
            masked_indices = masked_token_indices[source].unsqueeze(-1).expand(-1, -1, D)
            pred = pred.gather(1, masked_indices)
            v = v.gather(1, masked_indices)
            loss = nn.functional.mse_loss(pred, v, reduction="none")
            # Mask the loss where the tokens are not available
            m = m.gather(1, masked_indices).float()
            loss = (loss * m).sum() / m.sum()
            losses.append(loss)
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        """Defines a prediction step for the model.
        Returns:
            batch (dict of str to tuple of tensors): The input batch.
            pred (dict of str to tensor): The predicted values.
        """
        batch = self.preproc_input(batch)
        # Save the shapes of the padded tensors to reconstruct the images
        padded_shapes = {source: x[2].shape[-2:] for source, x in batch.items()}
        x = self.tokenize(batch)
        pred, masked_tokens_indices = self.forward(x)
        # For the tokens that were NOT masked, set them to the true values
        output = {}
        for source, (_, _, _, _, _, v, _) in x.items():
            otp = v.clone()
            B, L, D = pred[source].shape
            masked_indices = masked_tokens_indices[source].unsqueeze(-1).expand(-1, -1, D)
            otp.scatter_(1, masked_indices, pred[source])
            output[source] = otp
        # Convert the predictions back to images
        for source, v in output.items():
            pH, pW = padded_shapes[source]
            output[source] = patches_to_img(v, pH, pW, self.patch_size)
        return batch, output

    def configure_optimizers(self):
        """Configures the optimizer for the model.
        The optimizer is AdamW, with the default parameters.
        The learning rate goes through a linear warmup phase, then follows a cosine
        annealing schedule.
        """
        decay_params = {
            k: True for k, v in self.named_parameters() if "weight" in k and "norm" not in k
        }
        optimizer = torch.optim.AdamW(
            [
                {"params": [v for k, v in self.named_parameters() if k in decay_params]},
                {
                    "params": [v for k, v in self.named_parameters() if k not in decay_params],
                    "weight_decay": 0,
                },
            ]
        )

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            **self.lr_scheduler_kwargs,
        )
        return [optimizer], [scheduler]
