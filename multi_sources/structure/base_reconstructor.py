"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from multi_sources.models.embedding_layers import SourcetypeEmbedding0d, SourcetypeEmbedding2d
from multi_sources.models.output_layers import (
    SourcetypeProjection0d,
    SourcetypeProjection2d,
)

# Local module imports
from multi_sources.structure.base_module import MultisourceAbstractModule


class MultisourceAbstractReconstructor(MultisourceAbstractModule, ABC):
    """Given a torch model which receives inputs from multiple sources, this module masks
    some of the sources and trains its backbone to reconstruct the missing data.
    The sources may have different dimensionalities (e.g. 2D, 1D, 0D) and come with
    spatiotemporal coordinates.
    The structure expects its input as a dict D {(source_name, index): map}, where D[(source_name, index)] contains the
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

    The structure outputs a dict {(source_name, index): tensor} containing the predicted values
    for each source.
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
        loss_max_distance_from_center=None,
        ignore_land_pixels_in_loss=False,
        normalize_coords_across_sources=False,
        mask_only_sources=None,
        forecasting_mode=False,
        validation_dir=None,
        metrics={},
        use_modulation_in_output_layers=False,
        include_coords_in_conditioning=False,
        output_resnet_channels=None,
        output_resnet_blocks=None,
        sources_selection_seed=123,
        **kwargs,
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            backbone (nn.Module): Backbone model to train.
            n_sources_to_mask (int): Number of sources to mask in each sample.
            patch_size (int): Size of the patches to split the images into.
            values_dim (int): Dimension of the values embeddings.
            coords_dim (int): Dimension of the coordinates embeddings.
            adamw_kwargs (dict): Arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): Arguments to pass to the learning rate scheduler.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            ignore_land_pixels_in_loss (bool): If True, the pixels that are on land
                will be ignored in the loss computation.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            mask_only_sources (str or list of str): List of source types to mask. If not None,
                instead of randomly selecting the sources to mask, the model will always
                mask the sources of the specified types whenever they are available.
            forecasting_mode (bool): If True, will always mask all sources that are forecasted.
                A source is forecasted if its time delta is negative.
                Mutually exclusive with mask_only_sources.
            validation_dir (optional, str or Path): Directory where to save the validation plots.
                If None, no plots will be saved.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, y_true, masks, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
            use_modulation_in_output_layers (bool): If True, applies modulation to the output layers.
            include_coords_in_conditioning (bool): If True, includes the coordinates
                in the conditioning tensor used in the output layers.
            output_resnet_channels (int): Number of channels in the output ResNet.
            output_resnet_blocks (int): Number of blocks in the output ResNet.
            sources_selection_seed (int, optional): Seed for the random number generator used to select
                the sources to mask.
            **kwargs: Additional arguments to pass to the LightningModule constructor.
        """
        super().__init__(
            sources,
            cfg,
            adamw_kwargs,
            lr_scheduler_kwargs,
            loss_max_distance_from_center=loss_max_distance_from_center,
            ignore_land_pixels_in_loss=ignore_land_pixels_in_loss,
            normalize_coords_across_sources=normalize_coords_across_sources,
            validation_dir=validation_dir,
            metrics=metrics,
        )

        self.backbone = backbone
        self.n_sources_to_mask = n_sources_to_mask
        self.patch_size = patch_size
        self.values_dim = values_dim
        self.coords_dim = coords_dim

        # RNG that will be used to select the sources to mask
        self.source_select_gen = torch.Generator().manual_seed(sources_selection_seed)

        # Initialize the embedding layers
        self.init_embedding_layers(
            use_modulation_in_output_layers,
            include_coords_in_conditioning,
            output_resnet_channels,
            output_resnet_blocks,
        )

        if isinstance(mask_only_sources, str):
            mask_only_sources = [mask_only_sources]
        self.mask_only_sources = mask_only_sources
        # Convert from source types to the source names
        if self.mask_only_sources is not None:
            self.mask_only_sources = [
                src_name
                for src_name, src in self.sources.items()
                if src.type in self.mask_only_sources
            ]
        self.forecasting_mode = forecasting_mode

    def init_embedding_layers(
        self,
        use_modulation_in_output_layers,
        include_coords_in_conditioning,
        output_resnet_channels,
        output_resnet_blocks,
    ):
        """Initializes the weights of the embedding layers."""
        if not hasattr(self, "use_diffusion_t"):
            self.use_diffusion_t = False
        if not hasattr(self, "use_det_model"):
            self.use_det_model = False
        # Embedding and output projection layers
        # An embedding and an output layer for each source type
        # - We need to retrieve the list of each source type from the sources,
        #   as well as the number of characs variables for each source type.
        self.sourcetypes_characs_vars = {}
        self.sourcetype_embeddings = nn.ModuleDict()
        self.sourcetype_output_projs = nn.ModuleDict()
        for source in self.sources.values():
            # Only create the embedding layer for that source type if it doesn't exist yet
            if source.type not in self.sourcetypes_characs_vars:
                self.sourcetypes_characs_vars[source.type] = source.n_charac_variables()
                n_input_channels = source.n_input_variables()
                n_output_channels = source.n_output_variables()
                # Whether to include a predicted mean in the embedding layer
                pred_mean_channels = n_output_channels if self.use_det_model else 0
                # Create the layers for that source type depending on
                # its dimensionality
                if source.dim == 2:
                    self.sourcetype_embeddings[source.type] = SourcetypeEmbedding2d(
                        n_input_channels,
                        self.patch_size,
                        self.values_dim,
                        self.coords_dim,
                        source.n_charac_variables(),
                        use_diffusion_t=self.use_diffusion_t,
                        pred_mean_channels=pred_mean_channels,
                        include_coords_in_conditioning=include_coords_in_conditioning,
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection2d(
                        self.values_dim,
                        n_output_channels,
                        self.patch_size,
                        use_modulation=use_modulation_in_output_layers,
                        resnet_channels=output_resnet_channels,
                        resnet_blocks=output_resnet_blocks,
                    )
                elif source.dim == 0:
                    self.sourcetype_embeddings[source.type] = SourcetypeEmbedding0d(
                        n_input_channels,
                        self.values_dim,
                        self.coords_dim,
                        source.n_charac_variables(),
                        use_diffusion_t=self.use_diffusion_t,
                        use_predicted_mean=self.use_det_model,
                        pred_mean_channels=pred_mean_channels,
                    )
                    self.sourcetype_output_projs[source.type] = SourcetypeProjection0d(
                        self.values_dim,
                        n_output_channels,
                        use_modulation=use_modulation_in_output_layers,
                    )

            else:
                # Check that the number of characs variables is the same for all sources
                # of the same type
                if self.sourcetypes_characs_vars[source.type] != source.n_charac_variables():
                    raise ValueError(
                        "Number of characs variables is not "
                        "the same for all sources of type {source.type}"
                    )

    def embed(self, x):
        """Embeds the input sources. The embedded tensors' shapes depend on the dimensionality
        of the source:
        - For 2D sources: (B, h, w, Dv) for the values and (B, h, w, Dc) for the coordinates,
            where h = H // patch_size and w = W // patch_size.
        """
        output = {}
        for source_index_pair, data in x.items():
            source_name = source_index_pair[0]  # Extract the source name from the tuple
            source_obj = self.sources[source_name]
            # Only keep the current source's input variables from the values.
            input_mask = torch.tensor(
                source_obj.get_input_variables_mask(),
                device=data["values"].device,
                dtype=torch.bool,
            )
            emb_data = {k: v for k, v in data.items() if k != "values"}  # No in-place operation
            emb_data["values"] = data["values"][:, input_mask]

            source_type = source_obj.type
            v, c, cond = self.sourcetype_embeddings[source_type](emb_data)
            # v: embedded values, c: coords, cond: conditioning tensor

            output[source_index_pair] = {
                "embedded_values": v,
                "embedded_coords": c,
                "conditioning": cond,
                "avail": data["avail"].clone(),
            }

        return output

    def select_sources_to_mask(self, x):
        """Given a multi-sources batch, randomly selects a source to mask in each sample.
        Does not actually perform the masking.
        Args:
            x (dict of (source_name, index) to dict of str to tensor): The input sources.
        Returns:
            avail_flags (dict of (source_name, index) to tensor): The availability flags for each source,
                as tensors of shape (B,), such that:
                * avail_flags[s][i] == 1 if the source s is available for the sample i,
                * avail_flags[s][i] == 0 if the source s is masked for the sample i,
                * avail_flags[s][i] == -1 if the source s is missing for the sample i.
        """
        n_sources = len(x)
        any_elem = next(iter(x.values()))["values"]
        batch_size = any_elem.shape[0]
        device = any_elem.device

        if self.mask_only_sources is not None:
            # Case where there are pre-determined sources to mask. In this case,
            # we mask those whenever they are available.
            avail_flags = {}
            for source_index_pair, data in x.items():
                source_name = source_index_pair[0]  # Extract source name from tuple
                avail_flag = data["avail"].clone()
                if source_name in self.mask_only_sources:
                    avail_flag[avail_flag == 1] = 0
                avail_flags[source_index_pair] = avail_flag
            # We need to check that for each sample, at least one source has been masked.
            total_avail_flag = sum([flag == 0 for flag in avail_flags.values()])
            if (total_avail_flag == 0).any():
                raise ValueError(
                    "At least one sample has no sources to mask. "
                    "Please check the mask_only_sources argument."
                )
        # Case where we mask the sources that are forecasted.
        elif self.forecasting_mode:
            # In this case, we mask all sources that are forecasted (i.e. have a negative dt).
            avail_flags = {}
            for source_index_pair, data in x.items():
                source_name = source_index_pair[0]
                avail_flag = data["avail"].clone()
                # Mask the sources that are forecasted (i.e. have a negative dt).
                avail_flag[(avail_flag == 1) & (data["dt"] < 0)] = 0
                avail_flags[source_index_pair] = avail_flag
            # We need to check that for each sample, at least one source has been masked.
            total_avail_flag = sum([flag == 0 for flag in avail_flags.values()])
            if (total_avail_flag == 0).any():
                raise ValueError(
                    "At least one sample has no sources to mask. "
                    "Please check the forecasting_mode argument."
                )
        # Case where we randomly select the sources to mask
        else:
            # Select the sources to mask, which can differ between samples in the batch.
            # Missing sources cannot be masked.
            # Strategy: we'll generate a random noise tensor of shape (B, n_sources)
            # and for each row, mask the sources with the highest noise.
            noise = torch.rand((batch_size, n_sources), generator=self.source_select_gen).to(
                device
            )
            for i, (source_index_pair, data) in enumerate(x.items()):
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
            # Deduce the availability flags for each source
            avail_flags = {}
            for i, (source_index_pair, data) in enumerate(x.items()):
                avail_flag = data["avail"].clone()
                avail_flag[masked_sources_matrix[:, i]] = 0
                avail_flags[source_index_pair] = avail_flag
        return avail_flags

    @abstractmethod
    def mask(self, x, masking_seed=None):
        pass

    def forward(self, x, return_embeddings=False):
        """Computes the forward pass of the model.
        Args:
            x (dict of (source_name, index) to dict of str to tensor): The input sources, masked.
            return_embeddings (bool): If True, also returns the embedded values and coordinates.
        Returns:
            y (dict of (source_name, index) to tensor): The predicted values for each source.
            embeddings (optional, dict of (source_name, index) to dict of str to tensor):
                The embedded data for each source, if return_embeddings is True.
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
            # Embedded conditioning for the final modulation
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
        if return_embeddings:
            return pred, x
        return pred

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, input_batch, batch_idx):
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx):
        pass
