import lightning.pytorch as pl
import torch
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts


class SourcesReconstruction(pl.LightningModule):
    """Given a torch model which receives inputs from multiple sources, this class
    masks some of them and tries to reconstruct their values.
    The structure receives inputs as a map {source_name: (A, S, DT, C, D, V)}, where:
    - A is a scalar tensor of shape (1,) containing 1 if the element is available and -1 otherwise.
    - S is a scalar tensor of shape (1,) containing the index of the source.
    - DT is a scalar tensor of shape (1,) containing the time delta between the synoptic time
      and the element's time, normalized by dt_max.
    - C is a tensor of shape (3, H, W) containing the latitude, longitude, and land-sea mask.
    - D is a tensor of shape (H, W) containing the distance to the center of the storm.
    - V is a tensor of shape (n_variables, H, W) containing the variables for the source.

    Attributes:
        model (torch.nn.Module): The model to wrap.
    """

    def __init__(self, model, cfg, masked_sources, lr_scheduler_kwargs, metrics={}):
        """
        Args:
            model (torch.nn.Module): The model to wrap.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            masked_sources (list of str): The sources to mask during training.
            lr_scheduler_kwargs (dict): The keyword arguments to pass to the learning rate
                scheduler.
            metrics (dict of str: callable): The metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
        """
        super().__init__()
        self.model = model
        self.masked_sources = masked_sources
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.metrics = metrics
        self.save_hyperparameters(ignore="model")

    def loss_fn(self, y_pred, y_true):
        """Computes the reconstruction loss over the masked source(s).

        Args:
            y_pred (dict of str to tensor): The predicted values, as a dict
                {source_name: (S, DT, C, D, V)}.
            y_true (dict of str to tuple of tensors): The true values, as a
                dict {source_name: V}

        Returns:
            torch.Tensor: The loss of the model.
        """
        total_loss = 0
        for source_name, (_, _, c, d, v) in y_true.items():
            pred = y_pred[source_name]
            # Compute the MSE loss element-wise
            loss = (pred - v) ** 2
            # Ignore the pixels that are at least 1100km away from the center
            # of the storm.
            # This also ignores the pixels for which the distance is +inf, i.e. pixels
            # for which the target is not available.
            mask = (d <= 1100).unsqueeze(1).expand(loss.shape)
            masked_loss = loss[mask]
            if masked_loss.numel() == 0:
                continue
            total_loss += masked_loss.mean()

        return total_loss / len(y_true)

    def step(self, batch, batch_idx, train_or_val):
        """Defines a training or validation step for the model."""
        # Preprocess the input
        batch = self.preproc_input(batch)
        # Mask the sources
        input_sources, masked_sources = self.mask(batch)
        # Forward pass
        pred = self(input_sources, masked_sources)
        # Compute and log the loss
        loss = self.loss_fn(pred, masked_sources)
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
        input_sources, masked_sources = self.mask(batch)
        pred = self(input_sources, masked_sources)
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
            dt = torch.nan_to_num(dt, nan=1.0)
            v = torch.nan_to_num(v, nan=0)
            # Where the coords are NaN, set them to 90 for the latitude and 0
            # for the longitude (90 being an extreme value).
            c[:, 0] = torch.nan_to_num(c[:, 0], nan=90)
            c[:, 1] = torch.nan_to_num(c[:, 1], nan=0)
            # For the distance tensor, fill the nan values with +inf
            d = torch.nan_to_num(d, nan=float("inf"))
            # Normalize the lat/lon coordinates
            # Note: the lon is in the range [0, 360)
            c[:, 0] = c[:, 0] / 90
            c[:, 1] = (c[:, 1] - 180) / 180
            input_[source] = (a, s, dt, c, d, v)
        return input_

    def forward(self, input_sources, masked_sources):
        """Computes the forward pass of the model"""
        # The batches are given as dict {source_name: (A, S, DT, C, D, V)}
        # and {source_name: (S, DT, C, D)}, respectively.
        # but the model should receive two lists of (A, S, DT, C, PA, V) and
        # (S, DT, C) tensors.
        # PA means 'pixel availability' and is a boolean tensor indicating whether
        # each pixel is available or not in the source.
        input_batch = [
            (a, s, dt, c, (d < float("+inf")), v)
            for k, (a, s, dt, c, d, v) in input_sources.items()
        ]
        masked_batch = [(s, dt, c) for k, (s, dt, c, _, _) in masked_sources.items()]
        outputs = self.model(input_batch, masked_batch)
        # The model outputs a list of tensors, one for each masked source.
        # We need to map them back to the source names.
        return {k: v for k, v in zip(masked_sources.keys(), outputs)}

    def mask(self, x):
        """Given a training batch, creates two disjoint batches:
        - input_batch: Contains the sources that are not masked.
        - masked_batch: Contains the sources that are masked.

        Args:
            x (dict of str to tuple of tensors): The input batch.

        Returns:
            input_batch (dict of str to tuple of tensors): The input batch.
            masked_batch (dict of str to tuple of tensors): The masked batch.
        """
        input_sources, masked_sources = {}, {}
        for source, (a, s, dt, c, d, v) in x.items():
            if source in self.masked_sources:
                masked_sources[source] = (s, dt, c, d, v)
            else:
                input_sources[source] = (a, s, dt, c, d, v)
        return input_sources, masked_sources

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
