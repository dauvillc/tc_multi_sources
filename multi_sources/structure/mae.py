"""Implements the MultisourceMaskedAutoencoder class"""

import lightning.pytorch as pl
import torch
import multi_sources.metrics
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts


class MultisourceMaskedAutoencoder(pl.LightningModule):
    """Given a torch model which receives inputs from multiple sources, this class
    wraps it as a masked autoencoder, using Lightning.
    The structure receives inputs as a map {source_name: (A, S, DT, C, D, V)}, where:
    - A is a scalar tensor of shape (1,) containing 1 if the element is available and -1 otherwise.
    - S is a scalar tensor of shape (1,) containing the index of the source.
    - DT is a scalar tensor of shape (1,) containing the time delta between the synoptic time
      and the element's time, normalized by dt_max.
    - C is a tensor of shape (3, H, W) containing the latitude, longitude, and land-sea mask.
    - D is a tensor of shape (H, W) containing the distance to the center of the storm.
    - V is a tensor of shape (n_variables, H, W) containing the variables for the source.
    The model is expected to receive the same input map, and return a map {source_name: V'}
    where V' are the reconstructed values of the input.

    Attributes:
        model (torch.nn.Module): The model to wrap.
    """

    def __init__(self, model, lr_scheduler_kwargs, metrics={}, enable_masking=True,
                 only_mask_sources=[]):
        """
        Args:
            model (torch.nn.Module): The model to wrap.
            lr_scheduler_kwargs (dict): The keyword arguments to pass to the learning rate
                scheduler.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
            enable_masking (bool): Whether to enable masking of the input sources.
                Disable to run in the autoencoder mode.
            only_mask_sources (list of str): The sources to mask. If empty, all sources have
                an equal chance of being masked.
                If not empty, only the sources in the list can be masked.
        """
        super().__init__()
        self.model = model
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.metrics = metrics
        self.enable_masking = enable_masking
        self.only_mask_sources = only_mask_sources
        self.save_hyperparameters()

    def loss_fn(self, y_pred, y_true):
        """Computes the reconstruction loss over the masked source(s).
        The reconstruction loss is computed as the average over all pixels of:
          (ŷ - y)^2 * 1_{dist_to_center(pixel) < R} * (1 - land_mask(pixel)),
        where ŷ is the predicted value, y is the true value, and R is an arbitrary radius.
        Currently, R is chosen as 1100 km, which is enough to cover the largest storm
        ever recorded (Typhoon Tip, 1979, 2170 km diameter).

        Args:
            y_pred (dict): The predicted values. See the class description for the expected format.
            y_true (dict): The true values.

        Returns:
            torch.Tensor: The loss of the model.
        """
        total_loss = 0
        for source_name, (a, _, _, c, d, v) in y_true.items():
            pred = y_pred[source_name]
            # Compute the MSE loss element-wise
            loss = (pred - v) ** 2
            # Ignore all pixels for which d > 1100 km. Note that this also
            # excludes the pixels for which d = inf.= (padding)
            mask = (d <= 1100).unsqueeze(1).expand(loss.shape)
            # Ignore all pixels for which the land mask is 1
            mask = mask & (c[:, 2:3] == 0).expand(loss.shape)
            # Mask the loss for samples for which the source is not masked
            mask[a != 0] = False
            masked_loss = loss[mask]
            if masked_loss.numel() == 0:
                continue
            total_loss += masked_loss.mean()

        return total_loss / len(y_true)

    def step(self, batch, batch_idx, train_or_val):
        """Defines a training or validation step for the model."""
        # Preprocess the input
        batch = self.preproc_input(batch)
        # Forward pass
        pred = self(batch)
        # Compute and log the loss
        loss = self.loss_fn(pred, batch)
        self.log(f"{train_or_val}_loss", loss, prog_bar=True, on_epoch=True)
        # Compute metrics
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(pred, batch)
            self.log(f"{train_or_val}_{metric_name}", metric_value)
        return loss

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
        pred = self(batch)
        return batch, pred

    def preproc_input(self, x):
        """Fills NaN values in the input tensors with zeros, and normalizes the coordinates."""
        # x is a map {source_name: A, S, DT, C, D, V}
        # - Fill NaN values (masked / missing) in DT, C and V with zeros
        #   (which is also the mean value of the data after normalization)
        input_ = {}
        for source, (a, s, dt, c, d, v) in x.items():
            # Don't modify the tensors in-place, as we need to keep the NaN values
            # for the loss computation
            dt = torch.nan_to_num(dt, nan=0)
            c = torch.nan_to_num(c, nan=0)
            v = torch.nan_to_num(v, nan=0)
            # For the distance tensor, fill the nan values with +inf
            d = torch.nan_to_num(d, nan=float("inf"))
            # Normalize the lat/lon coordinates
            # Note: the lon is in the range [0, 360]
            c[:, 0] = c[:, 0] / 90
            c[:, 1] = (c[:, 1] - 180) / 180
            input_[source] = (a, s, dt, c, d, v)
        return input_

    def forward(self, batch):
        """Computes the forward pass of the model"""
        # Mask the input
        masked_batch = self.mask(batch)
        return self.model(masked_batch)

    def mask(self, x):
        """Given a training batch, randomly selects a source and masks all of its values.

        Args:
            x (dict of str to tuple of tensors): The input batch.

        Returns:
            masked_x (dict of str to tuple of tensors): The masked input batch.
        """
        if not self.enable_masking:
            return x
        # For each element in the batch, we need to select one source to mask.
        # However, we can't mask a source that is not available, as there is no
        # available target to compare the reconstruction with.
        # Small trick to do so:
        # - For each element, for each source, generate a random number between 0 and 1.
        # - If the source is not available, set the random number to -1.
        # - The source to mask is the one with the highest random number.
        # - If all sources are not available (random number is -1), raise an error.
        source_names = list(x.keys())
        batch_size = x[source_names[0]][0].shape[0]
        random_numbers = torch.rand(batch_size, len(source_names))
        for i, source_name in enumerate(source_names):
            a, _, _, _, _, _ = x[source_name]  # Availability tensor of shape (bs,)
            random_numbers[a == -1, i] = -1
            # If a set of maskable sources is provided, and this source is not in it,
            # set the random number to -1
            if self.only_mask_sources and source_name not in self.only_mask_sources:
                random_numbers[:, i] = -1
        values, indices = torch.max(random_numbers, dim=1)
        if values.min().item() == -1:
            raise ValueError("Found an element for which all sources are missing.")
        # Deduce which source to mask for each elemente
        masked_x = {}
        for i, (source_name, (a, s, dt, c, d, v)) in enumerate(x.items()):
            masked_v = v.clone()
            masked_v[indices == i] = 0
            # Set the availability tensor to 0 for the masked elements
            a[indices == i] = 0
            masked_x[source_name] = (a, s, dt, c, d, masked_v)

        return masked_x

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
