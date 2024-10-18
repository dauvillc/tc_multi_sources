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
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
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
    The structure outputs a dict {source_name: tensor} containing the predicted values for each source.
    """

    def __init__(
        self,
        source_names,
        backbone,
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
        use_attention_masks=False,
        output_convs=None,
        metrics={},
    ):
        """
        Args:
            source_names (list of str): Names of the sources.
            backbone (nn.Module): The backbone of the model.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            masking_ratio (float): The ratio of tokens to mask. If mask_full_sources is True,
                ratio of sources to mask instead.
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
            use_attention_masks (bool): If True, the model will use attention masks to
                ignore the tokens that are missing or masked.
            output_convs (dict of str to nn.Module, optional): Map from source name to
                a convolutional layer to apply to the output of the model.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
        """
        super().__init__()
        self.source_names = source_names
        self.backbone = backbone
        self.masking_ratio = masking_ratio
        self.patch_size = pair(patch_size)
        self.values_dim = values_dim
        self.metadata_dim = metadata_dim
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.adamw_kwargs = adamw_kwargs
        self.metrics = metrics
        self.loss_max_distance_from_center = loss_max_distance_from_center
        self.share_source_embeddings = share_source_embeddings
        self.use_attention_masks = use_attention_masks
        self.save_hyperparameters(ignore=["backbone", "metrics"])

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

        self.additional_values_info_norm = nn.LayerNorm(values_dim)
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
            # Deduce the availability mask level from where the values are missing
            # am = True where the values are available
            am = ~torch.isnan(v)[:, 0]  # (B, H, W)
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
            input_[source] = {
                "source_type": data["source_type"],
                "avail": data["avail"],
                "dt": dt,
                "context": ct,
                "coords": c,
                "landmask": lm,
                "dist_to_center": d,
                "values": v,
                "avail_mask": am,
            }
            # Check for NaN values in the tensors
            for k, v in input_[source].items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    if torch.isnan(v).any():
                        raise ValueError(f"NaN values found in source {source} tensor {k}")
        return input_

    def tokenize(self, x):
        """Converts the input sources into flattened sequences of tokens.
        Appends to the source dicts the key "tokens_shape" such that
        x[source]["tokens_shape"] = (Nw, Nh), where Nw and Nh are the number of tokens
        along the width and height of the image before flattening.
        """
        output = {}
        for source, data in x.items():
            # Compute the number of patches along the width and height
            h, w = data["coords"].shape[-2:]
            Nw = (w + self.patch_size[1] - 1) // self.patch_size[1]
            Nh = (h + self.patch_size[0] - 1) // self.patch_size[0]
            # c and v are image tensors of shape (b, c, h, w)
            c = img_to_patches(data["coords"], self.patch_size)
            v = img_to_patches(data["values"], self.patch_size)
            # lm, d and m are image tensors of shape (b, h, w)
            lm = img_to_patches(data["landmask"].unsqueeze(1), self.patch_size)
            d = img_to_patches(data["dist_to_center"].unsqueeze(1), self.patch_size)
            am = img_to_patches(data["avail_mask"].unsqueeze(1), self.patch_size)
            output[source] = {
                "source_type": data["source_type"],
                "tokens_shape": (Nw, Nh),
                "avail": data["avail"],
                "dt": data["dt"],
                "context": data["context"],
                "coords": c,
                "landmask": lm,
                "dist_to_center": d,
                "values": v,
                "avail_mask": am,
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
            am = self.mask_embedding(data["avail_mask"].float())

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

            output[source] = {
                "tokens_shape": data["tokens_shape"],
                "avail": data["avail"],
                "dt": dt,
                "embedded_coords": c,
                "embedded_source_meta": source_to_meta_embedding,
                "embedded_source_values": source_to_values_embedding,
                "embedded_dt": embedded_dt,
                "context": data["context"],
                "embedded_landmask": lm,
                "dist_to_center": data["dist_to_center"],
                "embedded_values": v,
                "embedded_avail_mask": am,
            }
        return output

    def mask(self, x):
        """Masks a portion of the sources. A missing source cannot be chosen to be masked.
        Supposes that there are at least as many non-missing sources as the number of sources
        to mask. The number of sources to mask is determined by self.masking ratio.
        When a source is masked, all of its values are replaced by the [MASK] token.
        Tokens from missing sources are also replaced by the [MASK] token, but their availability
        is left to -1 so that the loss is not computed for them.
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
                (avail_tensor == 0).view(B, 1, 1), self.mask_token.expand(B, L, D), values
            )
            masked_x[source]["embedded_values"] = masked_values
        return masked_x

    def sum_and_normalize(self, x):
        """Computes additional information about the values of the sources,
        which will be used within the backbone. Computes the metadata embeddings
        and normalizes them.
        Args:
            x (dict of str to dict of str to tensor): output of the embed or mask methods.
        Returns:
            x (dict of str to dict of str to tensor): Dict D such that D[source_name] contains
                the keys ['dt', 'embedded_dt', 'embedded_metadata', 'embedded_values', 'avail',
                'additional_values_info', 'tokens_shape'].
        """
        out = {}
        # The additional values info is the sum of:
        # - the embedded landmask
        # - the embedded availability mask
        # - the source-to-values embedding
        # - the embedded time delta.
        for source, data in x.items():
            out[source] = {
                "avail": data["avail"],
                "tokens_shape": data["tokens_shape"],
                "dt": data["dt"],
                "embedded_dt": data["embedded_dt"],
            }
            additional_values_info = (
                data["embedded_landmask"] + data["embedded_source_values"] + data["embedded_dt"]
            )
            # Where the source is not masked, add the embedded availability mask
            additional_values_info = torch.where(
                (data["avail"] == 1).view(-1, 1, 1),
                data["embedded_avail_mask"] + additional_values_info,
                additional_values_info,
            )
            # Normalize the additional values info and store it.
            additional_values_info = self.additional_values_info_norm(additional_values_info)
            out[source]["additional_values_info"] = additional_values_info
            # Compute the metadata embeddings from the time delta, the coordinates and
            # the source embeddings
            embedded_metadata = (
                data["embedded_coords"] + data["embedded_source_meta"] + data["embedded_dt"]
            )
            # Normalize the metadata and values embeddings
            out[source]["embedded_metadata"] = self.meta_norm(embedded_metadata)
            out[source]["embedded_values"] = self.values_norm(data["embedded_values"])
        return out

    def forward(self, x, padded_shapes):
        """Computes the forward pass of the model.
        Returns:
            pred (dict of str to tensor): The predicted values.
            avail_tensors (dict of str to tensor): The availability tensors for each source,
                of shape (B,), such that avail_tensors[source][b] is 1 if the source is available,
                0 if it was masked, and -1 if it was missing.
        """
        x = self.embed(x)
        # Mask a portion of the sources (replace the tokens by the [MASK] token)
        x = self.mask(x)
        avail_tensors = {source: data["avail"] for source, data in x.items()}
        # Compute the final values and metadata embeddings and normalize them
        x = self.sum_and_normalize(x)
        # If required, create an attention mask for each source that is either missing or masked.
        attn_masks = None
        if self.use_attention_masks:
            attn_masks = {}
            for source, data in x.items():
                B, L = data["embedded_values"].shape[:2]
                attn_mask = data["avail"] < 1  # (B,)
                attn_mask = attn_mask.view(B, 1).expand(B, L)
                attn_masks[source] = attn_mask
        # Forward x through the backbone
        pred = self.backbone(x, attn_masks)
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
        return pred, avail_tensors

    def step(self, batch, batch_idx, train_or_val):
        """Defines a training or validation step for the model."""
        batch = self.preproc_input(batch)
        batch_size = batch[list(batch.keys())[0]]["values"].shape[0]
        # Save the shapes of the padded tensors to reconstruct the images
        padded_shapes = {source: x["values"].shape[-2:] for source, x in batch.items()}

        x = self.tokenize(batch)
        pred, avail_tensors = self.forward(x, padded_shapes)

        # Compute the loss for each source
        losses = self.loss_fn(pred, x, avail_tensors)
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
            metric_value = metric_fn(pred, x)
            self.log(f"{train_or_val}_{metric_name}", metric_value, batch_size=batch_size)
        return loss

    def loss_fn(self, y_pred, y_true, avail_tensors):
        """Computes the MSE between the predicted and true values. The loss is only computed
        on the masked tokens. If a max distance from the center is specified, the loss is also
        only computed for the tokens that are within this distance from the center.
        Args:
            y_pred (dict of str to tensor): The predicted values.
            y_true (dict of str to dict of str to tensor): The unmasked input data.
            avail_tensors (dict of str to tensor): The availability tensors for each source.
        """
        losses = {}
        for source, true_data in y_true.items():
            # We'll compute a mask M on the tokens of the source of shape (B, L, D)
            # such that M[b, l, d] = True if and only if the following conditions are met:
            # - The source was masked for the sample b (avail_tensors[b] == 0);
            # - the value at position l, d was not missing (true_data["am"] == True);
            # - If self.loss_max_distance_from_center is not None, the token is within
            #   the specified distance from the center.
            B, L, D = true_data["values"].shape
            source_avail = avail_tensors[source].view(B, 1, 1).expand(B, L, D)  # (B, L, D)
            mask = (source_avail == 0) & true_data["avail_mask"]
            if self.loss_max_distance_from_center is not None:
                dist = true_data["dist_to_center"]
                mask = mask & (dist <= self.loss_max_distance_from_center)
            # Compute the loss
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
        batch = self.preproc_input(batch)
        # Save the shapes of the padded tensors to reconstruct the images
        padded_shapes = {source: x["values"].shape[-2:] for source, x in batch.items()}
        x = self.tokenize(batch)
        pred, avail_tensors = self.forward(x, padded_shapes)
        output = {}
        for source, true_data in x.items():
            output[source] = pred[source].clone()
        # Convert the predictions back to images
        for source, v in output.items():
            pH, pW = padded_shapes[source]
            output[source] = patches_to_img(v, pH, pW, self.patch_size)
        return batch, output, avail_tensors

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
