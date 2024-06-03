"""Implements the MultisourceMaskedAutoencoder class"""

import pytorch_lightning as pl
import torch
import multi_sources.metrics
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts


class MultisourceMaskedAutoencoder(pl.LightningModule):
    """Given a torch model which receives inputs from multiple sources, this class
    wraps it as a masked autoencoder, using Lightning.
    The structure receives inputs under the form of map {source_name: (S, DT, C, D, V)}, where:
    - S is a tensor of shape (batch_size,) containing the source index.
    - DT is a tensor of shape (batch_size,) containing the time delta between the element's time
        and the reference time.
    - C is a tensor of shape (batch_size, 3, H, W) containing the coordinates
        (lat, lon) of each pixel,
        and the land mask.
    - D is a tensor of shape (batch_size, H, W) containing the distance to the
        storm center at each pixel.
    - V is a tensor of shape (batch_size, channels, H, W) containing the values of each pixel.
    The model is expected to receive the same input map, and return a map {source_name: V'}
    where V' are the reconstructed values of the input.

    Attributes:
        model (torch.nn.Module): The model to wrap.
    """

    def __init__(self, model, lr_scheduler_kwargs, metrics={}):
        """
        Args:
            model (torch.nn.Module): The model to wrap.
            lr_scheduler_kwargs (dict): The keyword arguments to pass to the learning rate
                scheduler.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
        """
        super().__init__()
        self.model = model
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.metrics = metrics
        self.save_hyperparameters()

    def loss_fn(self, y_pred, y_true, masked_source_name):
        """Computes the reconstruction loss over the masked source.
        The reconstruction loss is computed as the average over all pixels of:
          (ŷ - y)^2 * 1_{dist_to_center(pixel) < R}
        where ŷ is the predicted value, y is the true value, and R is an arbitrary radius.
        Currently, R is chosen as 1100 km, which is enough to cover the largest storm
        ever recorded (Typhoon Tip, 1979, 2170 km diameter).

        Args:
            y_pred (dict): The predicted values. See the class description for the expected format.
            y_true (dict): The true values.
            masked_source_name (str): The name of the source that was masked (such
                that the task is to reconstruct this source only).

        Returns:
            torch.Tensor: The loss of the model.
        """
        _, _, _, d, v = y_true[masked_source_name]
        # Compute the MSE loss element-wise
        loss = (y_pred[masked_source_name] - v) ** 2
        # Mask the loss for pixels outside the storm radius
        # loss = loss * (d.unsqueeze(1) < 1100)
        return loss.mean()

    def step(self, batch, batch_idx, train_or_val):
        """Defines a training or validation step for the model."""
        # Preprocess the input
        batch = self.preproc_input(batch)
        # Forward pass
        pred, masked_source_name = self.forward(batch)
        # Compute and log the loss
        loss = self.loss_fn(pred, batch, masked_source_name)
        self.log(f"{train_or_val}_loss", loss, prog_bar=True)
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
        """Defines a prediction step for the model."""
        # Preprocess the input
        batch = self.preproc_input(batch)
        return self.forward(batch)[0]

    def preproc_input(self, x):
        """Fills NaN values in the input tensors with zeros, and normalizes the coordinates."""
        # x is a map {source_name: S, DT, C, D, V}
        # - Fill NaN values (masked / missing) in DT, C and V with zeros
        #   (which is also the mean value of the data after normalization)
        # - Remove D from the input, as it shouldn't be accessible to the model
        input_ = {}
        for source, (s, dt, c, d, v) in x.items():
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
            input_[source] = (s, dt, c, d, v)
        return input_

    def forward(self, batch):
        """Computes the forward pass of the model"""
        # Mask the input
        masked_batch, masked_source_name = self.mask(batch)
        # Remove the "distance to center" tensor from the input
        input_ = {source: (s, dt, c, v) for source, (s, dt, c, _, v) in masked_batch.items()}
        return self.model(input_), masked_source_name

    def mask(self, x):
        """Given a training batch, randomly selects a source and masks all of its values.

        Args:
            x (dict): The input batch. See the class description for the expected format.

        Returns:
            masked_x (dict): The masked input batch.
            masked_source_name (str): The name of the source that was masked.
        """
        # Randomly select a source to mask, until finding one such that
        # dt is not fully nan.
        source_name = None
        for source, (_, dt, _, _, _) in x.items():
            if not torch.isnan(dt).all():
                source_name = source
                break
        if source_name is None:
            raise ValueError("All sources are missing.")
        masked_x = {}
        for source, (s, dt, c, d, v) in x.items():
            if source == source_name:
                # Mask the values
                v = torch.zeros_like(v).to(v.dtype)
            masked_x[source] = (s, dt, c, d, v)
        return masked_x, source_name

    def configure_optimizers(self):
        """Configures the optimizer for the model.
        The optimizer is AdamW, with the default parameters.
        The learning rate goes through a linear warmup phase, then follows a cosine
        annealing schedule.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            **self.lr_scheduler_kwargs,
        )
        return [optimizer], [scheduler]

    def __call__(self, x):
        return self.forward(x)
