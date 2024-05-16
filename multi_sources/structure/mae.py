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
    - C is a tensor of shape (batch_size, 2, H, W) containing the coordinates (lat, lon) of each pixel.
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

    def loss_fn(self, y_pred, y_true):
        """Computes the loss of the model. The loss is defined as the mean squared error
        between the predicted and true values.

        Args:
            y_pred (dict): The predicted values. See the class description for the expected format.
            y_true (dict): The true values.

        Returns:
            torch.Tensor: The loss of the model.
        """
        loss = 0
        for source_name in y_pred:
            _, _, _, _, v = y_true[source_name]
            # Where y_true is masked (NaN), set the loss to zero
            mask = ~torch.isnan(v)
            loss += torch.mean((y_pred[source_name][mask] - v[mask])**2)

        return loss

    def training_step(self, batch, batch_idx):
        """Defines a training step for the model.

        Args:
            batch (dict): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the model.
        """
        #TODO: For now, we won't apply any masking, to just debug the pipeline.
        pred = self.forward(batch)
        loss = self.loss_fn(pred, batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Defines a validation step for the model.

        Args:
            batch (dict): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the model.
        """
        pred = self.model(batch)
        return self.loss_fn(pred, batch)

    def forward(self, x):
        # x is a map {source_name: S, DT, C, D, V}
        # - Fill NaN values (masked / missing) in DT, C and V with zeros
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

    def configure_optimizers(self):
        """Configures the optimizer for the model.
        The optimizer is AdamW, with the default parameters.
        The learning rate goes through a linear warmup phase, then follows a cosine
        annealing schedule.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=100, epochs=10)
        return [optimizer], [scheduler]
