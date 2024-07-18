"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import lightning.pytorch as pl
import torch
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts
from multi_sources.utils.image_processing import img_to_patches, pad_to_next_multiple_of, pair
from multi_sources.utils.image_processing import patches_to_img


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
    The structure gives to its backbone the inputs as a map {source_name: (S, DT, C, V, M)},
    where C and V have been tokenized to (batch, n_tokens, d) and:
    - M is a tensor of shape (batch, n_tokens, 1) containing 0 if the token is masked (or was
      missing) and 1 otherwise.
    """

    def __init__(self, model, cfg, masking_ratio, patch_size, lr_scheduler_kwargs, metrics={}):
        """
        Args:
            model (torch.nn.Module): The model to wrap.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            masking_ratio (float): The ratio of tokens to mask.
            patch_size (tuple of int): The size of the patches to split the images into.
            lr_scheduler_kwargs (dict): The arguments to pass to the learning rate scheduler.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
        """
        super().__init__()
        self.model = model
        self.masking_ratio = masking_ratio
        self.patch_size = pair(patch_size)
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.metrics = metrics
        self.save_hyperparameters(ignore="model")

    def preproc_input(self, x):
        # x is a map {source_name: A, S, DT, C, D, V}
        # - Discard A, which is not needed as its info can be inferred from C or D.
        # - Fill NaN values (masked / missing) in DT, C and V with zeros
        #   (which is also the mean value of the data after normalization)
        input_ = {}
        for source, (a, s, dt, c, d, v) in x.items():
            # Pad the image tensors to the next multiple of the patch size
            c = pad_to_next_multiple_of(c, self.patch_size, value=float("nan"))
            v = pad_to_next_multiple_of(v, self.patch_size, value=float("nan"))
            d = pad_to_next_multiple_of(d, self.patch_size, value=float("nan"))
            # C[:, 2:3] is the land mask, which we'll split from the latitude and longitude
            c, lm = c[:, :2], c[:, 2:3]
            # Don't modify the tensors in-place, as we need to keep the NaN values
            # for the loss computation
            dt = torch.nan_to_num(dt, nan=-1.0)
            v = torch.nan_to_num(v, nan=0)
            # Where the coords are NaN, set them to 90 for the latitude and 0
            # for the longitude (90 being an extreme value).
            c = torch.stack(
                (torch.nan_to_num(c[:, 0], nan=90), torch.nan_to_num(c[:, 1], nan=0)), dim=1
            )
            # For the distance tensor, fill the nan values with +inf
            d = torch.nan_to_num(d, nan=float("inf"))
            # Deduce the availability tensor from the distance tensor
            m = d != float("inf")
            input_[source] = (s, dt, c, d, lm, v, m)
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

    def mask(self, x):
        """Masks a portion of the tokens in the input.
        Returns a dict {source_name: s, dt, c, d, lm, v, m}
        where m is a tensor of shape (b, n_tokens, 1) whose value
        is 0 if the token is masked or missing, and 1 otherwise.
        """
        masked_sources, masked_and_avail = {}, {}
        for source, (s, dt, c, d, lm, v, m) in x.items():
            # Randomly mask a portion of the tokens in v
            mask = (torch.rand(v.shape[1]) < self.masking_ratio).to(v.device)
            masked_v = v.clone()
            masked_v[:, mask] = 0
            # At this point, m is as mask whose value is 0 if the token is missing
            # and 1 otherwise. We'll make it so that it's 0 if the token is masked
            # or missing, and 1 otherwise (i.e. for the model a missing token is
            # the same as a masked token).
            mask = mask.view(1, -1, 1).expand(m.shape)
            avail_for_model = (m & (~mask)).float()
            masked_sources[source] = (s, dt, c, d, lm, masked_v, avail_for_model)
            # For the loss function, we'll need a mask where the value is 1 if and only
            # if the token is masked and it was available.
            masked_and_avail[source] = m & mask
        return masked_sources, masked_and_avail

    def forward(self, x):
        # Right now, the model receives the inputs as a map {source_name: (s, dt, c, d, lm, v, m)}
        # But the model should receive a list of tuples (s, dt, c, lm, v, m)
        input = [(s, dt, c, lm, v, m) for s, dt, c, _, lm, v, m in x.values()]
        pred = self.model(input)
        # Recreate the map {source_name: pred}
        return {source: p for source, p in zip(x.keys(), pred)}

    def step(self, batch, batch_idx, train_or_val):
        """Defines a training or validation step for the model."""
        batch = self.preproc_input(batch)
        # Save the shapes after padding
        padded_shapes = {source: x[-1].shape[-2:] for source, x in batch.items()}
        # Tokenize, mask and predict
        x = self.tokenize(batch)
        x, loss_masks = self.mask(x)
        pred = self(x)
        # Convert the predictions back to the original shape
        for source, v in pred.items():
            pH, pW = padded_shapes[source]
            pred[source] = patches_to_img(v, pH, pW, self.patch_size)
            # Do the same on the loss masks
            loss_masks[source] = patches_to_img(loss_masks[source], pH, pW, self.patch_size)
        # Compute and log the loss
        loss = self.loss_fn(pred, batch, loss_masks)
        self.log(f"{train_or_val}_loss", loss, prog_bar=True, on_epoch=True)
        # Compute metrics
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(pred, batch)
            self.log(f"{train_or_val}_{metric_name}", metric_value)
        return loss

    def loss_fn(self, y_pred, y_true, loss_masks):
        """Computes the MSE between the predicted and true values."""
        losses = []
        for source, (_, _, _, _, _, v, m) in y_true.items():
            pred = y_pred[source]
            # Compute the loss only for the masked tokens
            loss = (pred - v) ** 2
            loss = loss[loss_masks[source]]
            losses.append(loss.mean())
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        """Defines a training step for the model.

        Args:
            batch (dict): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the model.
        """
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Defines a validation step for the model.

        Args:
            batch (dict): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the model.
        """
        return self.step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        """Defines a prediction step for the model.
        Returns:
            batch (dict of str to tuple of tensors): The input batch.
            pred (dict of str to tensor): The predicted values.
        """
        batch = self.preproc_input(batch)
        input_sources, masked_sources = self.mask(batch)
        pred = self(input_sources, masked_sources)
        return batch, pred

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
