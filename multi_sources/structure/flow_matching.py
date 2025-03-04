"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import lightning.pytorch as pl
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Flow matching imports
from flow_matching.path import CondOTProbPath

# Local module imports
from multi_sources.utils.scheduler import CosineAnnealingWarmupRestarts
from multi_sources.utils.solver import MultisourceEulerODESolver
from multi_sources.models.utils import (
    normalize_coords_across_sources,
    embed_coords_to_sincos,
)
from multi_sources.models.embedding_layers import (
    CoordinatesEmbedding2d,
    SourcetypeEmbedding2d,
    CoordinatesEmbedding0d,
    SourceSpecificEmbedding0d,
)
from multi_sources.models.output_layers import (
    SourcetypeProjection0d,
    SourcetypeProjection2d,
)

# Visualization imports
from multi_sources.utils.visualization import display_solution_html


class MultisourceFlowMatchingReconstructor(pl.LightningModule):
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.
    The structure expects its input as a dict {source_name: map}, where each map contains the
    following key-value pairs (all shapes excluding the batch dimension):
    - "avail" is a scalar tensor of shape (B,) containing 1 if the element is available
        and -1 otherwise.
    - "values" is a tensor of shape (B, C, ...) containing the values of the source.
    - "landmask" is a tensor of shape (B, ...) containing the land-sea mask of the source.
    - "coords" is a tensor of shape (B, 2, ...) containing the spatial coordinates of the source.
    - "dt" is a scalar tensor of shape (B,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "dist_to_center" is a tensor of shape (B, ...) containing the distance
        to the center of the storm at each pixel, in km.
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
        n_sampling_diffusion_steps,
        adamw_kwargs,
        lr_scheduler_kwargs,
        loss_max_distance_from_center=None,
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
            n_sampling_diffusion_steps (int): Number of diffusion steps when sampling.
            adamw_kwargs (dict): Arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): Arguments to pass to the learning rate scheduler.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            validation_dir (optional, str or Path): Directory where to save the validation plots.
                If None, no plots will be saved.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true) -> torch.Tensor.
            **kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__()
        self.sources = {source.name: source for source in sources}
        self.source_names = [source.name for source in sources]
        self.backbone = backbone
        self.n_sources_to_mask = n_sources_to_mask
        self.patch_size = patch_size
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.n_sampling_diffusion_steps = n_sampling_diffusion_steps
        self.validation_dir = validation_dir
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.loss_max_distance_from_center = loss_max_distance_from_center
        self.adamw_kwargs = adamw_kwargs
        self.metrics = metrics
        self.normalize_coords_across_sources = normalize_coords_across_sources

        # Save the configuration so that it can be loaded from the checkpoints
        self.cfg = cfg
        self.save_hyperparameters(ignore=["backbone", "metrics"])

        # Embedding and output projection layers
        # An embedding and an output layer for each source type
        # - We need to retrieve the list of each source type from the sources,
        #   as well as the number of characs variables for each source type.
        self.sourcetypes_characs_vars = {}
        self.sourcetype_embeddings = nn.ModuleDict()
        self.sourcetype_output_projs = nn.ModuleDict()
        self.sourcetype_coords_embeddings = nn.ModuleDict()
        self.source_to_type = {source.name: source.type for source in sources}
        for source in sources:
            # Only create the embedding layer for that source type if it doesn't exist yet
            if source.type not in self.sourcetypes_characs_vars:
                self.sourcetypes_characs_vars[source.type] = source.n_charac_variables()
                n_output_channels = source.n_data_variables()
                # Create the layers for that source type depending on
                # its dimensionality
                if source.dim == 2:
                    self.sourcetype_embeddings[source.type] = SourcetypeEmbedding2d(
                        source.n_data_variables(),
                        self.patch_size,
                        values_dim,
                    )
                    self.sourcetype_coords_embeddings[source.type] = CoordinatesEmbedding2d(
                        self.patch_size,
                        coords_dim,
                        source.n_charac_variables(),
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection2d(
                        self.values_dim, self.coords_dim, n_output_channels, self.patch_size
                    )
                elif source.dim == 0:
                    self.sourcetype_embeddings[source.type] = SourceSpecificEmbedding0d(
                        source.n_data_variables(),
                        values_dim,
                    )
                    self.sourcetype_coords_embeddings[source.type] = CoordinatesEmbedding0d(
                        coords_dim,
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection0d(
                        self.values_dim, self.coords_dim, n_output_channels
                    )
            else:
                # Check that the number of characs variables is the same for all sources
                # of the same type
                if self.sourcetypes_characs_vars[source.type] != source.n_charac_variables():
                    raise ValueError(
                        f"Number of characs variables is not "
                        "the same for all sources of type {source.type}"
                    )

        # Token that will be added to the embedded values of the masked sources only,
        # to help the model distinguish between the masked and unmasked tokens.
        # Flow matching ingredients
        self.fm_path = CondOTProbPath()
        self.cnt = 0

    def preproc_input(self, x):
        # Normalize the coordinates across sources to make them relative instead of absolute
        # (i.e the min coord across all sources of a sample is always 0 and the max is 1).
        coords = [source["coords"] for source in x.values()]
        normed_coords = embed_coords_to_sincos(coords)
        if self.normalize_coords_across_sources:
            normed_coords = normalize_coords_across_sources(normed_coords)

        input_ = {}
        for i, (source, data) in enumerate(x.items()):
            c, v = normed_coords[i].float(), data["values"].float()
            lm, d = data["landmask"].float(), data["dist_to_center"].float()
            dt = data["dt"].float()
            # Deduce the availability mask level from where the values are missing
            # am = True where the values are available
            # /!\ The availability mask is different from the avail flag. The avail flag
            # is a single value for the whole source, which is -1 if the source is missing,
            # and 1 if it's available. The availability mask gives the availability of each
            # point in the source: 1 if the point is available, 0 if it's masked, -1 if missing.
            am = (~torch.isnan(v)[:, 0]).float() * 2 - 1  # (B, H, W)
            # Don't modify the tensors in-place, as we need to keep the NaN values
            # for the loss computation
            dt = torch.nan_to_num(data["dt"], nan=-1.0)
            v = torch.nan_to_num(v, nan=0)
            lm = torch.nan_to_num(lm, nan=0)
            # Where the coords are NaN, set them to -1, as the normalization set the non-nan values
            # to [0, 1]
            c = torch.nan_to_num(c, nan=-1)
            # For the distance tensor, fill the nan values with +inf
            d = torch.nan_to_num(d, nan=float("inf"))
            # Potential characs variables
            if "characs" in data:
                ch = data["characs"].float()
                ch = torch.nan_to_num(ch, nan=0)

            # Create two separate dictionaries: one for embedding input, one for loss computation
            embed_input = {
                "avail": data["avail"],
                "dt": dt,
                "coords": c,
                "values": v,
                "characs": ch if "characs" in data else None,
            }

            loss_info = {
                "avail_mask": am,
                "landmask": lm,
                "dist_to_center": d,
            }

            input_[source] = {
                **embed_input,  # Data needed for embedding
                **loss_info,  # Additional data needed for loss computation
            }
        return input_

    def embed(self, x):
        """Embeds the input sources."""
        output = {}
        for source, data in x.items():
            # Embed the source's values
            source_type = self.source_to_type[source]
            v = self.sourcetype_embeddings[source_type](data)
            # Embed the source's coordinates
            c = self.sourcetype_coords_embeddings[source_type](data)
            # Save the layout of the tokens for the output projection. For example
            # a tokens_shape of (3, 2) means 3 tokens in the first dimension and 2 in the second.
            # That info is lost in the embedded sequences as they are flattened.
            if len(data["values"].shape) > 2:
                tokens_shape = tuple(
                    int(np.ceil(s / self.patch_size)) for s in data["values"].shape[2:]
                )
            else:
                # For 0D sources, the tokens shape is (1,)
                tokens_shape = (1,)

            output[source] = {
                "tokens_shape": tokens_shape,
                "embedded_values": v,
                "embedded_coords": c,
            }

        return output

    def mask(self, x, target_source=None, pure_noise=False):
        """Masks a portion of the sources. A missing source cannot be chosen to be masked.
        Supposes that there are at least as many non-missing sources as the number of sources
        to mask. The number of sources to mask is determined by self.masking ratio.
        Masked sources have their values mixed with random noise following the noise schedule.
        The availability flag is set to 0 where the source is masked.
        Args:
            x (dict of str to dict of str to tensor): The input sources.
            target_source (optional, str): If specified, the source to mask. If None, the source
                to mask is chosen randomly for each sample in the batch.
            pure_noise (bool): If True, the sources are masked with pure noise, without
                following the noise schedule.
        Returns:
            masked_x (dict of str to dict of str to tensor): The input sources with a portion
                of the sources masked. An entry "diffusion_t" is added to the dict of each source,
                which is a tensor of shape (B,) such that: diffusion_t[b] is the diffusion timestep
                at which the source was masked for the sample b if the source was masked,
                and -1 otherwise.
            path_samples (dict of str to PathSample): the ProbSample objects used to generate the
                noised values.
        """
        n_sources = len(x)
        any_elem = next(iter(x.values()))["values"]
        batch_size = any_elem.shape[0]
        device = any_elem.device

        if target_source is None:
            # Select the sources to mask, which can differ between samples in the batch.
            # Missing sources cannot be masked.
            # Strategy: we'll generate a random noise tensor of shape (B, n_sources)
            # and for each row, mask the sources with the highest noise.
            noise = torch.rand((batch_size, n_sources), device=device)
            for i, (source, data) in enumerate(x.items()):
                # Multiply the noise by the availability mask (-1 for missing sources, 1 otherwise)
                noise[:, i] = noise[:, i] * data["avail"].squeeze(-1)
            # Gather the indices of the sources to mask for each sample
            _, sources_to_mask = noise.topk(
                self.n_sources_to_mask, dim=1
            )  # (B, n_sources_to_mask)
            # Deduce a matrix M of shape (B, n_sources) such that M[b, i] = 1 if the source i
            # should be masked for the sample b, and 0 otherwise.
            masked_sources_matrix = torch.zeros(
                (batch_size, n_sources), dtype=torch.bool, device=device
            )  # (B, n_sources)
            masked_sources_matrix.scatter_(1, sources_to_mask, True)
        else:  # If a target source is specified, mask it for all samples in the batch
            if target_source not in x:
                raise ValueError(f"Source {target_source} is missing from the input")
            masked_sources_matrix = torch.zeros(
                (batch_size, n_sources), dtype=torch.bool, device=device
            )
            for i, source in enumerate(x):
                if source == target_source:
                    masked_sources_matrix[:, i] = True

        masked_x, path_samples = {}, {}
        for i, (source, data) in enumerate(x.items()):
            # Copy the data to avoid modifying the original dict
            masked_data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
            # Set the avail flag to 0 where the source should be masked
            avail_flag = masked_data["avail"]
            avail_flag[masked_sources_matrix[:, i]] = 0
            # The sources should be noised if they are masked.
            should_noise = avail_flag == 0
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
            noised_values = path_sample.x_t.detach().clone()
            masked_data["values"][should_noise] = noised_values[should_noise]
            # Set the availability mask to 0 everywhere for noised sources.
            # (!= from the avail flag, it's a mask of same shape
            masked_data["avail_mask"][should_noise] = -1
            # Save the diffusion timesteps at which the source was masked. For unnoised sources,
            # the diffusion step is set to 1, since the step t=1 means no noise is left.
            masked_data["diffusion_t"] = torch.where(should_noise, t, torch.ones_like(t))
            # For sources that are fully unavailable, set the diffusion timestep to -1
            masked_data["diffusion_t"][masked_data["avail"] == -1] = -1
            path_samples[source] = path_sample
            masked_x[source] = masked_data
        return masked_x, path_samples

    def forward(self, x):
        """Computes the velocity fields for a given input.
        Args:
            x (dict of str to dict of str to tensor): The input sources, already noised.
        Returns:
            pred_vf (dict of str to tensor): The predicted velocity fields for each source.
        """
        # Save the shape of the tokens before they're embedded, so that we can
        # later remove the padding.
        spatial_shapes = {
            source: data["values"].shape[2:]
            for source, data in x.items()
            if len(data["values"].shape) > 2
        }
        # Embed and mask the sources
        x = self.embed(x)

        # Run the transformer backbone
        pred = self.backbone(x)

        # No need to project the other sources if we're fine-tuning
        for source, v in pred.items():
            # Embedded coords for the final modulation
            c = x[source]["embedded_coords"]
            # Project from latent values space to output space using the output layer
            # corresponding to the source type
            pred[source] = self.sourcetype_output_projs[self.source_to_type[source]](
                v, c, tokens_shape=x[source]["tokens_shape"]
            )
        # For 2D sources, remove the padding
        for source, spatial_shape in spatial_shapes.items():
            pred[source] = pred[source][..., : spatial_shape[0], : spatial_shape[1]]
        return pred

    def mask_and_loss_step(self, batch, batch_idx, train_or_val):
        """Given a preprocessed batch of samples, masks a portion of the sources, and
        computes the conditional flow matching loss."""
        batch_size = batch[list(batch.keys())[0]]["values"].shape[0]

        # Mask the sources (noising)
        masked_x, path_samples = self.mask(batch)
        # Make predictions
        pred = self.forward(masked_x)
        # Retrieve the availability flag for each source updated after masking
        avail_flag = {source: data["avail"] for source, data in masked_x.items()}

        losses = self.loss_fn(path_samples, pred, batch, avail_flag)
        # If len(losses) == 0, i.e. for all masked sources the tokens were missing,
        # raise an error.
        if len(losses) == 0:
            raise ValueError("No tokens to compute the loss on")
        # Compute the total loss
        loss = sum(losses.values()) / len(losses)

        self.log(
            f"{train_or_val}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
        )
        return loss

    def loss_fn(self, path_samples, pred, y_true, avail_flag):
        """Computes the conditional flow matching loss for the model. Only pixels
        that were noised are considered in the loss computation.
        Variables that have been tagged as input-only in their source are not considered
        in the loss computation.
        Args:
            path_samples (dict of str to PathSample): The ProbSample objects used to generate
                the noised values.
            pred (dict of str to tensor): The predictions for each source.
            y_true (dict of str to dict of str to tensor): Input data, with the following
                keys:
                - "avail_mask": The availability mask of shape (B, ...).
                - "dist_to_center": The distance to the center of the storm of shape (B, ...).
            avail_flag (dict of str to tensor): The availability flag of each source, as tensor
                of shape (B,) containing 1 if the element is available, 0 if it was masked and
                -1 if it was missing.
        """
        losses = {}
        for source, true_data in y_true.items():
            pred_s = pred[source]
            # Retrieve the mask of the output-enabled variables for the source,
            # and exclude them from the predicted and true velocity fields.
            output_vars = self.sources[source].get_output_variables_mask()  # (C,)
            output_vars = torch.tensor(output_vars, device=pred_s.device)
            pred_s = pred_s[:, output_vars]
            target_s = path_samples[source].dx_t[:, output_vars]
            # debug: imshow x_t, the pred and the target
            # if self.cnt != 0 and self.cnt % 10000 == 0:
            #     for i in range(pred_s.shape[0]):
            #         if avail_flag[source][i] == 0:
            #             fig = plt.figure()
            #             ax1 = fig.add_subplot(131)
            #             ax2 = fig.add_subplot(132)
            #             ax3 = fig.add_subplot(133)
            #             ax1.imshow(path_samples[source].x_t[i, 0].detach().cpu().numpy())
            #             ax2.imshow(pred_s[i, 0].detach().cpu().numpy())
            #             ax3.imshow(target_s[i, 0].detach().cpu().numpy())
            #             plt.show()
            #             plt.close()
            # self.cnt += 1

            # We'll compute a mask M on the tokens of the source of shape (B, C, ...)
            # such that M[b, ...] = True if and only if the following conditions are met:
            # - The source was masked for the sample b (avail flag == 0);
            # - the value at position ... was not missing (true_data["am"] == True);
            loss_mask = true_data["avail_mask"] >= 1  # (B, ...)
            loss_mask[avail_flag[source] != 0] = False
            # If a maximum distance from the center is specified, exclude the pixels
            # that are too far from the center from the loss computation.
            if self.loss_max_distance_from_center is not None:
                dist_mask = true_data["dist_to_center"] <= self.loss_max_distance_from_center
                loss_mask = loss_mask & dist_mask
            # Expand the mask to the number of channels in the source
            loss_mask = loss_mask.unsqueeze(1).expand_as(target_s)  # (B, C, ...)
            target_s = target_s[loss_mask]
            pred_s = pred_s[loss_mask]
            if target_s.numel() == 0:
                continue
            loss = (pred_s - target_s).pow(2).mean()
            losses[source] = loss
        return losses

    def sample(self, batch):
        """Samples the model using multiple steps of the ODE solver. All sources
        that have an availability flag set to 0 or -1 are solved.
        Args:
            batch (dict of str to dict of str to tensor): The input batch, preprocessed.
        Returns:
            time_grid (torch.Tensor): The time grid at which the ODE solver sampled the solution,
                of shape (T,).
            sol (dict of str to torch.Tensor): The solution of the ODE solver for each source,
                as tensors of shape (T, B, C, ...).
        """
        # Mask the sources with pure noise
        masked_batch, path_samples = self.mask(batch, pure_noise=True)
        x_0 = {source: data["values"] for source, data in masked_batch.items()}  # pure noise

        def vf_func(x_t, t):
            """Function that computes the velocity fields of each source
            included in x."""
            # Don't modify masked_x in-place
            batch_t = {
                source: {k: v for k, v in data.items()} for source, data in masked_batch.items()
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
        sol = solver.solve(x_0, time_grid)

        return time_grid, sol

    def training_step(self, batch, batch_idx):
        batch = self.preproc_input(batch)
        return self.mask_and_loss_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        batch = self.preproc_input(batch)
        if self.validation_dir is not None and batch_idx % 5 == 0:
            # Sample the model for each source in the batch
            time_grid, sol = self.sample(batch)
            # Create a visualization in HTML for 4 evenly spaced samples in the batch
            batch_size = next(iter(batch.values()))["values"].shape[0]
            # Calculate indices for 4 evenly spaced samples
            sample_indices = np.linspace(0, batch_size - 1, 4, dtype=int)
            for i in sample_indices:
                fig = display_solution_html(batch, sol, time_grid, sample_index=i)
                fig.write_html(self.validation_dir / f"sample_{batch_idx}_{i}.html")
        return self.mask_and_loss_step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        """Defines a prediction step for the model.
        Returns:
            batch (dict of str to dict of str to tensor): The input batch.
            pred (dict of str to tensor): The predicted values.
            avail_tensors (dict of str to tensor): The availability tensors for each source.
        """
        # TODO
        pass

    def configure_optimizers(self):
        """Configures the optimizer for the model.
        The optimizer is AdamW, with the default parameters.
        The learning rate goes through a linear warmup phase, then follows a cosine
        annealing schedule.
        """
        decay = self.adamw_kwargs.pop("weight_decay", 0.0)
        params = {k: v for k, v in self.named_parameters() if v.requires_grad}

        # Apply weight decay only to the weights that are not in the normalization layers
        decay_params = {k for k, _ in params.items() if "weight" in k and "norm" not in k}
        optimizer = torch.optim.AdamW(
            [
                # Parameters without decay
                {"params": [v for k, v in params.items() if k not in decay_params]},
                # Parameters with decay
                {
                    "params": [v for k, v in params.items() if k in decay_params],
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
