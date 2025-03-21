"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import lightning.pytorch as pl
import torch
import torch.nn as nn

# Flow matching imports
from flow_matching.path import CondOTProbPath

# Local module imports
from multi_sources.structure.base_module import MultisourceAbstractReconstructor
from multi_sources.utils.solver import MultisourceEulerODESolver

# Visualization imports
from multi_sources.utils.visualization import display_realizations


class MultisourceFlowMatchingReconstructor(MultisourceAbstractReconstructor):
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
        predict_residuals=False,
        pretrained_det=None,
        n_sampling_diffusion_steps=25,
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
            predict_residuals (bool): If True, the model will use a deterministic model
                to predict the mean, and the FM to predict the residuals.
            pretrained_det (str or Path): Path to a pretrained deterministic model.
                Can only be used if predict_residuals is True.
            n_sampling_diffusion_steps (int): Number of diffusion steps when sampling.
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
            **kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        self.use_diffusion_t = True
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
        # Flow matching ingredients
        self.n_sampling_diffusion_steps = n_sampling_diffusion_steps
        self.fm_path = CondOTProbPath()

    def mask(self, x, pure_noise=False, masking_seed=None):
        """Masks a portion of the sources. A missing source cannot be chosen to be masked.
        Supposes that there are at least as many non-missing sources as the number of sources
        to mask. The number of sources to mask is determined by self.masking ratio.
        Masked sources have their values mixed with random noise following the noise schedule.
        The availability flag is set to 0 where the source is masked.
        Args:
            x (dict of str to dict of str to tensor): The input sources.
            pure_noise (bool): If True, the sources are masked with pure noise, without
                following the noise schedule.
            masking_seed (int, optional): Seed for the random number generator used to select
                which sources to mask.
        Returns:
            masked_x (dict of str to dict of str to tensor): The input sources with a portion
                of the sources masked. If not determnistic, an entry "diffusion_t" is added
                to the dict of each source, which is a tensor of shape (B,) such that:
                diffusion_t[b] is the diffusion timestep at which the source was masked
                for the sample b if the source was masked, and -1 otherwise.
            path_samples (dict of str to PathSample): the ProbSample objects used to generate the
                noised values.
        """

        # Choose the sources to mask.
        avail_flags = super().select_sources_to_mask(x, masking_seed)
        # avail_flags[s][i] == 0 if the source s should be masked.
        batch_size = avail_flags[list(avail_flags.keys())[0]].shape[0]
        device = next(self.parameters()).device

        masked_x, path_samples = {}, {}
        for source, data in x.items():
            # Copy the data to avoid modifying the original dict
            masked_data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
            # Update the availability flag to that after masking.
            masked_data["avail"] = avail_flags[source]
            # The sources should be noised if they are masked.
            should_mask = avail_flags[source] == 0

            # NOISING
            if pure_noise:
                t = torch.zeros(batch_size, device=device)  # Means x_t = x_0
            else:
                # Sigmoid of a standard gaussian to favor intermediate timesteps,
                # following Stable Diffusion 3.
                t = torch.normal(mean=0, std=1, size=(batch_size,), device=device).sigmoid()
            # Generate random noise with the same shape as the values
            noise = torch.randn_like(masked_data["values"])
            # Compute the noised values associated with the diffusion timesteps
            path_sample = self.fm_path.sample(t=t, x_0=noise, x_1=data["values"].detach().clone())
            path_samples[source] = path_sample
            noised_values = path_sample.x_t.detach().clone()
            masked_data["values"][should_mask] = noised_values[should_mask]
            # Save the diffusion timesteps at which the source was masked. For unnoised sources,
            # the diffusion step is set to 1, since the step t=1 means no noise is left.
            masked_data["diffusion_t"] = torch.where(should_mask, t, torch.ones_like(t))
            # For sources that are fully unavailable, set the diffusion timestep to -1
            masked_data["diffusion_t"][masked_data["avail"] == -1] = -1
            # Set the availability mask to 0 everywhere for noised sources.
            # (!= from the avail flag, it's a mask of same shape
            masked_data["avail_mask"][should_mask] = 0
            masked_x[source] = masked_data

        return masked_x, path_samples

    def compute_loss(self, pred, batch, masked_batch, path_samples):
        # Retrieve the availability flag for each source updated after masking
        avail_flag = {source: data["avail"] for source, data in masked_batch.items()}

        # Filter the predictions and true values
        # For FM, the true values are the velocity fields
        true_y = {source: path_samples[source].dx_t for source in path_samples}
        avail_mask = {source: batch[source]["avail_mask"] for source in batch}
        dist_to_center = {source: batch[source]["dist_to_center"] for source in batch}
        landmask = {source: batch[source]["landmask"] for source in batch}
        pred, true_y = super().apply_loss_mask(
            pred, true_y, avail_flag, avail_mask, dist_to_center, landmask
        )

        # Compute the loss: MSE between the predicted and true velocity fields
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

    def sample(self, batch, n_realizations_per_sample):
        """Samples the model using multiple steps of the ODE solver. All sources
        that have an availability flag set to 0 or -1 are solved.
        Args:
            batch (dict of str to dict of str to tensor): The input batch, preprocessed.
            n_realizations_per_sample (int): Number Np of realizations to sample for each
                element in the batch.
        Returns:
            avail_flags (dict of str to tensor): The availability flags for each source,
                after masking, as tensors of shape (B,).
            time_grid (torch.Tensor): The time grid at which the ODE solver sampled the solution,
                of shape (T,).
            sol (dict of str to torch.Tensor): The solution of the ODE solver for each source,
                as tensors of shape (Np, B, C, ...).
        """
        with torch.no_grad():
            # We'll use the same seed for the selection of the sources to mask,
            # so that the realizations are consistent between samples and only the noise
            # differs.
            seed = torch.Generator().seed()
            all_sols = []  # Will store each realization of the solution
            for _ in range(n_realizations_per_sample):
                # Mask the sources with pure noise
                masked_batch, path_samples = self.mask(batch, pure_noise=True, masking_seed=seed)

                x_0 = {
                    source: data["values"] for source, data in masked_batch.items()
                }  # pure noise

                def vf_func(x_t, t):
                    """Function that computes the velocity fields of each source
                    included in x."""
                    # Don't modify masked_x in-place
                    batch_t = {
                        source: {k: v for k, v in data.items()}
                        for source, data in masked_batch.items()
                    }
                    # Update the values and diffusion timesteps of the sources that are solved.
                    for source, x_ts in x_t.items():
                        is_solved = batch_t[source]["avail"] == 0
                        batch_t[source]["values"][is_solved] = x_ts[is_solved]
                        batch_t[source]["diffusion_t"][is_solved] = t
                    # Run the model
                    vf = self.forward(batch_t)
                    # Where the sources are not being solved, we'll set the velocity field to zero,
                    # so that those examples don't change in the solution.
                    for source in vf:
                        vf[source][batch_t[source]["avail"] != 0] = 0
                    return vf

                # Solve the ODE
                time_grid = torch.linspace(0, 1, self.n_sampling_diffusion_steps)
                solver = MultisourceEulerODESolver(vf_func)
                sol = solver.solve(x_0, time_grid)  # Dict of tensors of shape (B, C, ...)
                all_sols.append(sol)

            avail_flags = {source: data["avail"] for source, data in masked_batch.items()}
            all_sols = {source: torch.stack([sol[source] for sol in all_sols]) for source in batch}
            return avail_flags, time_grid, all_sols

    def training_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]]["values"].shape[0]
        batch = self.preproc_input(batch)
        # Mask the sources
        masked_x, path_samples = self.mask(batch)
        # Make predictions
        pred = self.forward(masked_x)
        # Compute the loss
        loss = self.compute_loss(pred, batch, masked_x, path_samples)

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
        masked_x, path_samples = self.mask(batch)
        # Make predictions
        pred = self.forward(masked_x)
        # Compute the loss
        loss = self.compute_loss(pred, batch, masked_x, path_samples)
        self.log(
            f"val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if self.validation_dir is not None and batch_idx % 5 == 0:
            # For every 10 batches, make multiple realizations of the solution
            # for each sample in the batch and display them.
            if batch_idx % 10 == 0:
                avail_flags, time_grid, sol = self.sample(batch, n_realizations_per_sample=5)
                display_realizations(
                    sol,
                    input_batch,
                    avail_flags,
                    self.validation_dir / f"realizations_{batch_idx}",
                )
            else:
                # We'll only compute the metrics on one realization of the solution.
                avail_flags, time_grid, sol = self.sample(batch, n_realizations_per_sample=1)
            # Only keep one realization of the solution for the metrics.
            sol = {source: sol[source][0] for source in sol}
            # Evaluate the metrics
            for metric_name, metric in self.metrics.items():
                metric_res = metric(sol, batch, avail_flags)
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
        """Defines a prediction step for the model.
        Returns:
            batch (dict of str to dict of str to tensor): The input batch.
            pred (dict of str to tensor): The predicted values.
            avail_tensors (dict of str to tensor): The availability tensors for each source.
        """
        # TODO
        pass


def load_model(checkpoint_path):
    """Loads the lightning module from the checkpoint.
    Args:
        checkpoint_path (str or Path): Path to the checkpoint to load.
    Returns:
        model (MultisourceFlowMatchingReconstructor): The loaded model.
    """
    return MultisourceFlowMatchingReconstructor.load_from_checkpoint(checkpoint_path)
