"""Implements the MultisourceMaskedAutoencoder class"""
import pytorch_lightning as pl
import torch


class MultisourceMaskedAutoencoder(pl.LightningModule):
    """Given a torch model which receives inputs from multiple sources, this class
    wraps it as a masked autoencoder, using Lightning.
    The structure receives inputs under the form of map {source_name: (S, DT, C, D, V)}, where:
    - S is a tensor of shape (batch_size,) containing the source index.
    - DT is a tensor of shape (batch_size,) containing the time delta between the element's time
        and the reference time.
    - C is a tensor of shape (batch_size, 3, H, W) containing the coordinates (lat, lon) of each pixel,
        and the land mask.
    - D is a tensor of shape (batch_size, H, W) containing the distance to the storm center at each pixel.
    - V is a tensor of shape (batch_size, channels, H, W) containing the values of each pixel.
    The model is expected to receive the same input map, and return a map {source_name: V'} where V'
    are the reconstructed values of the input.

    Attributes:
        model (torch.nn.Module): The model to wrap.
    """
    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): The model to wrap.
        """
        super().__init__()
        self.model = model

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
        loss = loss * (d.unsqueeze(1) < 1100)
        # Set the loss to zero for NaN values in V
        loss[torch.isnan(v)] = 0
        return loss.mean()

    def training_step(self, batch, batch_idx):
        """Defines a training step for the model.

        Args:
            batch (dict): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the model.
        """
        # Randomly select a source and mask its values
        masked_batch, masked_source_name = self.mask(batch)
        pred = self.forward(masked_batch)
        loss = self.loss_fn(pred, batch, masked_source_name)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Defines a validation step for the model.

        Args:
            batch (dict): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the model.
        """
        masked_batch, masked_source_name = self.mask(batch)
        pred = self.forward(masked_batch)
        loss = self.loss_fn(pred, batch, masked_source_name)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def forward(self, x):
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
            input_[source] = (s, dt, c, v)
        return self.model(input_)

    def mask(self, x):
        """Given a training batch, randomly selects a source and masks all of its values.
        
        Args:
            x (dict): The input batch. See the class description for the expected format.

        Returns:
            masked_x (dict): The masked input batch.
            masked_source_name (str): The name of the source that was masked.
        """
        # Select a source to mask, until finding a source whose values are not all NaN
        source_name = None
        perm = torch.randperm(len(x))
        keys = list(x.keys())
        while source_name is None:
            source_name = keys[perm[0]]
            _, _, _, _, v = x[source_name]
            if torch.isnan(v).all():
                source_name = None
        if source_name is None:
            raise ValueError('All sources have NaN values, cannot use any of them as training target.')
        masked_x = {}
        for source, (s, dt, c, d, v) in x.items():
            if source == source_name:
                # Mask the values
                v = torch.full_like(v, float('nan')).to(v.dtype)
            masked_x[source] = (s, dt, c, d, v)
        return masked_x, source_name

    def configure_optimizers(self):
        """Configures the optimizer for the model.
        The optimizer is AdamW, with the default parameters.
        The learning rate goes through a linear warmup phase, then follows a cosine
        annealing schedule.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=100, epochs=10)
        return [optimizer], [scheduler]

    def __call__(self, x):
        return self.forward(x)
