"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

from pathlib import Path

import torch

# Flow matching imports
from flow_matching.path import CondOTProbPath
from hydra.utils import instantiate

# Local module imports
from multi_sources.structure.base_reconstructor import MultisourceAbstractReconstructor
from multi_sources.utils.solver import MultisourceEulerODESolver

# Visualization imports
from multi_sources.utils.visualization import display_realizations
from utils.checkpoints import load_experiment_cfg_from_checkpoint


class MultisourceFlowMatchingReconstructor(MultisourceAbstractReconstructor):
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.
    The structure expects its input as a dict D {(source_name, index): map},
    where D[(source_name, index)] contains the following key-value pairs
    (all shapes excluding the batch dimension):
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

    The structure outputs a dict {(source_name, index): tensor} containing
    the predicted values for each source.
    """

    def __init__(
        self,
        sources,
        cfg,
        backbone,
        n_sources_to_mask,
        patch_size,
        values_dim,
        coords_dim,
        adamw_kwargs,
        lr_scheduler_kwargs,
        use_det_model_from_run=None,
        n_sampling_diffusion_steps=25,
        loss_max_distance_from_center=None,
        ignore_land_pixels_in_loss=False,
        normalize_coords_across_sources=False,
        validation_dir=None,
        compute_metrics_every_k_batches=5,
        display_realizations_every_k_batches=3,
        n_realizations_per_sample=5,
        metrics={},
        use_modulation_in_output_layers=False,
        det_model_kwargs={},
        cfg_train_uncond_proba=0.0,
        cfg_scale=None,
        **kwargs,
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            backbone (torch.nn.Module): Multi-sources backbone model.
            n_sources_to_mask (int): Number of sources to mask in each sample.
            patch_size (int): Size of the patches to split the images into.
            values_dim (int): Dimension of the values embeddings.
            coords_dim (int): Dimension of the coordinates embeddings.
            adamw_kwargs (dict): Arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): Arguments to pass to the learning rate scheduler.
            use_det_model_from_run (str): Path to a checkpoint of a deterministic model
                to predict the means from.
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
            compute_metrics_every_k_batches (int): Number of batches between two metric computations,
                which require sampling with the ODE solver.
            display_realizations_every_kp_batches (int): Number of metrics evaluations between
                two realizations display.
            n_realizations_per_sample (int): Number of realizations to sample for each sample
                during validation or testing.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true, masks, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
            use_modulation_in_output_layers (bool): If True, the output layers will apply
                modulation to the values embeddings before projecting them to the output space.
            det_model_kwargs (dict): Arguments to pass to the deterministic model
                constructor, if using a deterministic model.
            cfg_train_uncond_proba (float): Probability of dropping the other sources
                when training, for classifier-free guidance.
            cfg_scale (float): Guidance scale for classifier-free guidance.
                If None, no guidance is applied.
            **kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        self.use_diffusion_t = True
        self.use_det_model = use_det_model_from_run is not None
        super().__init__(
            sources,
            cfg,
            backbone,
            n_sources_to_mask,
            patch_size,
            values_dim,
            coords_dim,
            adamw_kwargs,
            lr_scheduler_kwargs,
            loss_max_distance_from_center=loss_max_distance_from_center,
            ignore_land_pixels_in_loss=ignore_land_pixels_in_loss,
            normalize_coords_across_sources=normalize_coords_across_sources,
            validation_dir=validation_dir,
            metrics=metrics,
            use_modulation_in_output_layers=use_modulation_in_output_layers,
            **kwargs,
        )

        # Flow matching ingredients
        self.n_sampling_diffusion_steps = n_sampling_diffusion_steps
        self.fm_path = CondOTProbPath()
        self.compute_metrics_every_k_batches = compute_metrics_every_k_batches
        self.display_realizations_every_k_batches = display_realizations_every_k_batches
        self.n_realizations_per_sample = n_realizations_per_sample
        self.cfg_train_uncond_proba = cfg_train_uncond_proba
        self.cfg_scale = cfg_scale

        # Optional: deterministic model usage
        if self.use_det_model:
            ckpt_dir = Path(cfg["paths"]["checkpoints"])
            # Load the checkpoint and the configuration of the deterministic model
            det_cfg, ckpt_path = load_experiment_cfg_from_checkpoint(
                ckpt_dir, use_det_model_from_run
            )
            self.det_model = instantiate(
                det_cfg["lightning_module"],
                sources,
                det_cfg,
                validation_dir=validation_dir,
                **det_model_kwargs,
            )
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
            self.det_model.load_state_dict(state_dict, strict=True)
            self.det_model.eval()
            self.det_model.requires_grad_(False)

    def mask(self, x, pure_noise=False, masking_seed=None):
        """Masks a portion of the sources. A missing source cannot be chosen to be masked.
        Supposes that there are at least as many non-missing sources as the number of sources
        to mask. The number of sources to mask is determined by self.masking ratio.
        Masked sources have their values mixed with random noise following the noise schedule.
        The availability flag is set to 0 where the source is masked.

        If using a deterministic model, the masked sources are converted to the residual
        between the ground truth and the predicted mean, before being noised.

        Args:
            x (dict): The input sources with (source_name, index) tuples as keys,
                where index counts observations (0 = most recent).
            pure_noise (bool): If True, the sources are masked with pure noise, without
                following the noise schedule.
            masking_seed (int, optional): Seed for the random number generator used to select
                which sources to mask.
        Returns:
            masked_x (dict): The input sources with a portion
                of the sources masked. An entry "diffusion_t" is added
                to the dict of each source, which is a tensor of shape (B,) such that:
                diffusion_t[b] is the diffusion timestep at which the source was masked
                for the sample b if the source was masked, and -1 otherwise.
            path_samples (dict): the ProbSample objects used to generate the
                noised values.
        """

        # First step: for each sample in the batch, select a subset of the sources to mask.
        avail_flags = super().select_sources_to_mask(x, masking_seed)
        # avail_flags[s][i] == 0 if the source s should be masked.
        device = next(self.parameters()).device

        # Second step: create a copy of the input dict and update the availability flag
        # to 0 for the masked sources.
        masked_x = {}
        for source_index_pair, data in x.items():
            # Copy the data to avoid modifying the original dict
            masked_data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
            # Update the availability flag to that after masking.
            masked_data["avail"] = avail_flags[source_index_pair]
            # Set the availability mask to 0 everywhere for noised sources.
            masked_data["avail_mask"][masked_data["avail"] == 0] = 0
            masked_x[source_index_pair] = masked_data

        # Third step (optional): if using a deterministic model, compute the predicted means
        # and convert the masked sources to the residual between the ground truth and the predicted
        # mean.
        if self.use_det_model:
            # Compute the deterministic predictions (which are set to 0 for unmasked sources)
            masked_x = self.make_deterministic_predictions(masked_x)
            # Convert the target values to the residuals.
            for source_index_pair, data in masked_x.items():
                pred_mean = masked_x[source_index_pair]["pred_mean"]
                # Note: pred_mean is set to 0 for unmasked sources, so no change in this case.
                masked_x[source_index_pair]["values"] -= pred_mean

        # Last step: for each masked source, compute the noised values
        # and the diffusion timestep at which the source was masked.
        path_samples = {}
        for source_index_pair, masked_data in masked_x.items():
            # NOISING
            batch_size = masked_data["values"].shape[0]
            if pure_noise:
                t = torch.zeros(batch_size, device=device)  # Means x_t = x_0
            else:
                t = torch.rand(batch_size, device=device)  # Random diffusion timestep
            # Generate random noise with the same shape as the values
            noise = torch.randn_like(masked_data["values"], device=device)
            # The sources should be noised if they are masked.
            should_mask = avail_flags[source_index_pair] == 0
            # Compute the noised values associated with the diffusion timesteps
            path_sample = self.fm_path.sample(
                t=t, x_0=noise, x_1=masked_data["values"].detach().clone()
            )
            path_samples[source_index_pair] = path_sample
            noised_values = path_sample.x_t.detach().clone()
            masked_data["values"][should_mask] = noised_values[should_mask]
            # Save the diffusion timesteps at which the source was masked. For unnoised sources,
            # the diffusion step is set to 1, since the step t=1 means no noise is left.
            masked_data["diffusion_t"] = torch.where(should_mask, t, torch.ones_like(t))
            # For sources that are fully unavailable, set the diffusion timestep to -1
            masked_data["diffusion_t"][masked_data["avail"] == -1] = -1
            masked_x[source_index_pair] = masked_data

        return masked_x, path_samples

    def forward(self, x):
        """Computes the forward pass of the model.
        Args:
            x (dict): The input sources, masked, with (source_name, index) tuples as keys.
        Returns:
            y (dict): The predicted values for each source. If using CFG, the output
                tensors will be double batches, where the first and second halves
                correspond to the original batch and the unconditional copy, respectively.
        """
        # Save the shape of the tokens before they're embedded, so that we can
        # later remove the padding.
        spatial_shapes = {
            source_index_pair: data["values"].shape[2:]
            for source_index_pair, data in x.items()
            if len(data["values"].shape) > 2
        }
        # Embed and mask the sources
        x = self.embed(x)

        # Run the transformer backbone
        pred = self.backbone(x)

        for source_index_pair, v in pred.items():
            # Embedded condtioning for the final modulation.
            cond = x[source_index_pair]["conditioning"]
            # Project from latent values space to output space using the output layer
            # corresponding to the source type
            source_name = source_index_pair[0]  # Extract source name from tuple
            src_type = self.sources[source_name].type
            pred[source_index_pair] = self.sourcetype_output_projs[src_type](v, cond)
        # For 2D sources, remove the padding
        for source_index_pair, spatial_shape in spatial_shapes.items():
            pred[source_index_pair] = pred[source_index_pair][
                ..., : spatial_shape[0], : spatial_shape[1]
            ]
        return pred

    def compute_loss(self, pred, batch, masked_batch, path_samples):
        # Retrieve the availability flag for each source updated after masking
        avail_flag = {
            source_index_pair: data["avail"] for source_index_pair, data in masked_batch.items()
        }

        # Filter the predictions and true values
        # For FM, the true values are the velocity fields
        y_true = {
            source_index_pair: path_samples[source_index_pair].dx_t
            for source_index_pair in path_samples
        }

        # Only the keep the output variables from the ground truth
        y_true = self.filter_output_variables(y_true)
        # Compute the loss masks: a dict {(s,i): M} where M is a binary mask of shape
        # (B, ...) indicating which points should be considered in the loss.
        loss_masks = self.compute_loss_mask(batch, avail_flag)

        # Compute the MSE between the true and predicted velocity fields loss for each source
        losses = {}
        for source_index_pair in pred:
            # Compute the pointwise loss for each source.
            source_loss = (pred[source_index_pair] - y_true[source_index_pair]).pow(2)
            # Multiply by the loss mask
            source_loss_mask = loss_masks[source_index_pair].unsqueeze(1).expand_as(source_loss)
            source_loss = source_loss * source_loss_mask
            # Compute the mean over the number of available points
            mask_sum = source_loss_mask.sum()
            if mask_sum == 0:
                # If all points are masked, we skip the loss computation for this source
                continue
            losses[source_index_pair] = source_loss.sum() / mask_sum

        # Compute the total loss
        loss = sum(losses.values()) / len(losses)
        return loss

    def sample(self, batch, n_realizations_per_sample):
        """Samples the model using multiple steps of the ODE solver. All sources
        that have an availability flag set to 0 or -1 are solved.
        Args:
            batch (dict): The input batch, preprocessed, with (source_name, index) tuples as keys.
            n_realizations_per_sample (int): Number Np of realizations to sample for each
                element in the batch.
        Returns:
            dict: A dictionary containing the following keys:
                - avail_flags (dict): The availability flags for each source,
                    after masking, as tensors of shape (B,).
                - time_grid (torch.Tensor): The time grid at which the ODE solver sampled the solution,
                    of shape (T,).
                - sol (dict): The solution of the ODE solver for each source,
                    as tensors of shape (Np, B, C, ...).
                - pred_mean (dict): If using a deterministic model, the predicted means
                    for each source, as tensors of shape (B, C, ...).
        """
        with torch.no_grad():
            # We'll use the same seed for the selection of the sources to mask,
            # so that the realizations are consistent between samples and only the noise
            # differs.
            seed = torch.Generator().seed()
            all_sols = []  # Will store each realization of the solution
            for _ in range(n_realizations_per_sample):
                # Mask the sources with pure noise. If using CFG, also generate an unconditional version
                # of the samples in a double batch.
                masked_batch, path_samples = self.mask(batch, pure_noise=True, masking_seed=seed)

                x_0 = {
                    source_index_pair: data["values"]
                    for source_index_pair, data in masked_batch.items()
                }  # pure noise

                def vf_func(x_t, t):
                    """Function that computes the velocity fields of each source
                    included in x."""
                    # Don't modify masked_x in-place
                    batch_t = {
                        source_index_pair: {k: v for k, v in data.items()}
                        for source_index_pair, data in masked_batch.items()
                    }
                    # Update the values and diffusion timesteps of the sources that are solved.
                    for source_index_pair, x_ts in x_t.items():
                        is_solved = batch_t[source_index_pair]["avail"] == 0
                        batch_t[source_index_pair]["values"][is_solved] = x_ts[is_solved]
                        batch_t[source_index_pair]["diffusion_t"][is_solved] = t
                    # Run the model
                    if self.cfg_scale is not None:
                        # If using classifier-free guidance (CFG), we need to create a double batch
                        # where the first half contains the original batch and the second half
                        # contains an unconditional copy of the batch.
                        vf = self.forward(self.to_double_batch(batch_t))
                        # Split the conditional and unconditional batches and apply the
                        # guidance scale.
                        if self.cfg_scale is not None:
                            half_batch = next(iter(vf.values())).shape[0] // 2
                            for source_index_pair in vf:
                                cond = vf[source_index_pair][:half_batch]
                                uncond = vf[source_index_pair][half_batch:]
                                vf[source_index_pair] = cond + self.cfg_scale * (cond - uncond)
                    else:
                        vf = self.forward(batch_t)
                    # Where the sources are not being solved, we'll set the velocity field to zero,
                    # so that those examples don't change in the solution.
                    for source_index_pair in vf:
                        vf[source_index_pair][batch_t[source_index_pair]["avail"] != 0] = 0

                    return vf

                # Solve the ODE
                time_grid = torch.linspace(0, 1, self.n_sampling_diffusion_steps)
                solver = MultisourceEulerODESolver(vf_func)
                sol = solver.solve(x_0, time_grid)  # Dict of tensors of shape (B, C, ...)

                # If using a deterministic model, the solution of the ODE is the residual
                # between the predicted mean and the actual sample. We need to add back the
                # predicted mean to the solution.
                if self.use_det_model:
                    for source_index_pair in sol:
                        # Note: the predicted mean is already set to 0 for unmasked sources.
                        pred_mean = masked_batch[source_index_pair]["pred_mean"]
                        sol[source_index_pair] += pred_mean

                all_sols.append(sol)

            avail_flags = {
                source_index_pair: data["avail"]
                for source_index_pair, data in masked_batch.items()
            }
            all_sols = {
                source_index_pair: torch.stack([sol[source_index_pair] for sol in all_sols])
                for source_index_pair in batch
            }

            returned_dict = {
                "avail_flags": avail_flags,
                "time_grid": time_grid,
                "sol": all_sols,
            }
            # If using a deterministic model, also return the predicted means
            if self.use_det_model:
                returned_dict["pred_mean"] = {
                    source_index_pair: masked_batch[source_index_pair]["pred_mean"]
                    for source_index_pair in masked_batch
                }
            return returned_dict

    def make_deterministic_predictions(self, masked_x):
        """Computes the deterministic predictions of the model.
        Args:
            masked_x (dict): The input sources, masked, with (source_name, index) tuples as keys.
        Returns:
            masked_x (dict): The input sources with the predicted values
                for each source, included in the dict under
                updated_x[source_idx_pair]["pred_mean"].
                The dict is updated in-place.
        """
        # Run the deterministic model
        means = self.det_model(masked_x)
        # Update the values of the sources with the predicted values
        for source_index_pair, data in masked_x.items():
            pred_mean = means[source_index_pair]
            # The predicted means are only valid for the masked sources
            # (i.e. the sources that have an availability flag set to 0).
            # -> Set the predicted means to 0 for the sources that are not masked.
            pred_mean[data["avail"] != 0] = 0
            # Update the values with the predicted values
            masked_x[source_index_pair]["pred_mean"] = pred_mean
        return masked_x

    def training_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]]["values"].shape[0]
        batch = self.preproc_input(batch)
        masked_x, path_samples = self.mask(batch)

        # Classifier-free guidance (CFG) training
        if self.cfg_train_uncond_proba > 0.0:
            # Randomly select a subset of the batch to make unconditional
            any_tensor = next(iter(masked_x.values()))["values"]
            batch_size, device = any_tensor.shape[0], any_tensor.device
            which_samples = torch.rand(batch_size, device=device) < self.cfg_train_uncond_proba
            masked_x = self.to_unconditional_batch(masked_x, which_samples=which_samples)

        # Make predictions
        pred = self.forward(masked_x)
        # Compute the loss
        loss = self.compute_loss(pred, batch, masked_x, path_samples)

        self.log(
            "train_loss",
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
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if self.validation_dir is not None:
            if batch_idx % self.compute_metrics_every_k_batches == 0:
                # We'll sample a certain number of realizations for each sample.
                # If we're displaying realizations, we'll sample 5 realizations. If we're
                # simply computing the metrics, we'll sample 1 realization.
                if batch_idx % self.display_realizations_every_k_batches == 0:
                    n_real = self.n_realizations_per_sample
                else:
                    n_real = 1
                # Sample with the ODE solver
                sampling_dict = self.sample(batch, n_realizations_per_sample=n_real)
                sol, avail_flags = sampling_dict["sol"], sampling_dict["avail_flags"]

                # If required, display the realizations
                if batch_idx % self.display_realizations_every_k_batches == 0:
                    display_realizations(
                        sol,
                        input_batch,
                        avail_flags,
                        self.validation_dir / f"realizations_{batch_idx}",
                        display_fraction=1.0,
                    )

                # Only keep one realization of the solution for the metrics.
                sol = {source: sol[source][0] for source in sol}
                # Evaluate the metrics
                y_true = {
                    source_index_pair: batch[source_index_pair]["values"]
                    for source_index_pair in batch
                }
                y_true = self.filter_output_variables(y_true)
                masks = self.compute_loss_mask(batch, avail_flags)
                for metric_name, metric in self.metrics.items():
                    metric_res = metric(sol, y_true, masks)
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
        """Samples with the ODE solver and returns the predicted values
        for each source, as well as the availability flags after masking.
        Returns:
            outputs (dict): A dictionary containing:
                - sol (dict): The predicted values for each source,
                  as tensors of shape (R, B, C, ...),
                  where R is the number of realizations sampled.
                - avail_flags (dict): The availability flags for each source,
                  after masking, as tensors of shape (B,).
                - pred_mean (dict, optional): If using a deterministic model,
                  the predicted means for each source, as tensors of shape (B, C, ...).
        """
        batch = self.preproc_input(batch)

        # Sample with the ODE solver
        sampling_dict = self.sample(
            batch, n_realizations_per_sample=self.n_realizations_per_sample
        )

        sol, avail_flags = sampling_dict["sol"], sampling_dict["avail_flags"]

        # Create output dictionary with solutions and availability flags
        output = {"sol": sol, "avail_flags": avail_flags}

        # If using a deterministic model, add the predicted means as a separate key in the output
        if self.use_det_model and "pred_mean" in sampling_dict:
            output["pred_mean"] = sampling_dict["pred_mean"]

        return output

    def to_unconditional_batch(self, batch, which_samples=None):
        """Given a batch where some of the sources are masked, creates an unconditional
        copy of the batch where the unmasked sources are erased.
        Extends the base class method to handle flow matching specific fields.
        Args:
            batch (dict of (source_name, index) to dict): The input batch. Expects
                the same structure as is output by self.preproc_input().
            which_samples (torch.Tensor, optional): If provided, boolean tensor of shape (B,)
                indicating which samples in the batch should be made unconditional. The others
                will be left unchanged. If None, all samples will be made unconditional.
        Returns:
            unconditional_batch (dict of (source_name, index) to dict): The unconditional copy
                of the batch, where the unmasked sources are erased.
                The structure is the same as the input batch.
        """
        # First get the base unconditional batch
        unconditional_batch = super().to_unconditional_batch(batch, which_samples)

        # Add flow matching specific fields
        for source_index_pair, data in batch.items():
            where_avail = data["avail"] == 1  # (B,)
            if which_samples is not None:
                where_avail = where_avail & which_samples

            # For the unconditional copy, set the diffusion timestep to -1
            # for the unmasked sources (as in self.mask()).
            unconditional_batch[source_index_pair]["diffusion_t"] = torch.where(
                where_avail, -1.0, data["diffusion_t"]
            )

            # If using deterministic model, handle predicted means
            if self.use_det_model and "pred_mean" in data:
                # Set the predicted means to 0 for the unmasked sources
                unconditional_batch[source_index_pair]["pred_mean"] = torch.where(
                    where_avail.view((-1,) + (1,) * (data["pred_mean"].ndim - 1)),
                    torch.zeros_like(data["pred_mean"]),
                    data["pred_mean"],
                )

        return unconditional_batch

    def to_double_batch(self, batch, which_samples=None):
        """Given a batch where some of the sources are masked, creates a double batch
        where the first half contains the original batch and the second half contains
        an unconditional copy of the batch, where the unmasked sources are erased.
        This is used for classifier-free guidance (CFG) training.
        Args:
            batch (dict of (source_name, index) to dict): The input batch. Expects
                the same structure as is output by self.preproc_input().
            which_samples (torch.Tensor, optional): If provided, boolean tensor of shape (B,)
                indicating which samples in the batch should be made unconditional. The others
                will be left unchanged. If None, all samples will be made unconditional.
        Returns:
            double_batch (dict of (source_name, index) to dict): The double batch.
        """
        # Create the unconditional copy of the batch
        unconditional_batch = self.to_unconditional_batch(batch, which_samples)

        # Concatenate the original batch and the unconditional copy along the batch dimension
        double_batch = {}
        for source_index_pair, data in batch.items():
            new_data = unconditional_batch[source_index_pair]
            double_batch[source_index_pair] = {
                "avail": torch.cat([data["avail"], new_data["avail"]], dim=0),
                "dt": torch.cat([data["dt"], new_data["dt"]], dim=0),
                "coords": torch.cat([data["coords"], new_data["coords"]], dim=0),
                "values": torch.cat([data["values"], new_data["values"]], dim=0),
                "landmask": torch.cat([data["landmask"], new_data["landmask"]], dim=0),
                "avail_mask": torch.cat([data["avail_mask"], new_data["avail_mask"]], dim=0),
                "dist_to_center": torch.cat(
                    [data["dist_to_center"], new_data["dist_to_center"]], dim=0
                ),
                "diffusion_t": torch.cat([data["diffusion_t"], new_data["diffusion_t"]], dim=0),
            }
            if "characs" in data:
                double_batch[source_index_pair]["characs"] = torch.cat(
                    [data["characs"], new_data["characs"]], dim=0
                )
            if self.use_det_model and "pred_mean" in data:
                double_batch[source_index_pair]["pred_mean"] = torch.cat(
                    [data["pred_mean"], new_data["pred_mean"]], dim=0
                )

        return double_batch


def load_model(checkpoint_path):
    """Loads the lightning module from the checkpoint.
    Args:
        checkpoint_path (str or Path): Path to the checkpoint to load.
    Returns:
        model (MultisourceFlowMatchingReconstructor): The loaded model.
    """
    return MultisourceFlowMatchingReconstructor.load_from_checkpoint(checkpoint_path)
