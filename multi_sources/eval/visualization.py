"""
Implements visual evaluation comparison for multi-source predictions.

This evaluation class creates visualization figures showing targets and predictions
for each sample. The figures are organized as follows:
- Top row: Available sources (avail=1) and target source s(avail=0)
- Subsequent rows: One row per model showing predictions for each source

Usage in Hydra config:
evaluation_classes:
  visual:
    _target_: 'multi_sources.eval.visualization_new.VisualEvaluationComparison'
    eval_fraction: 1.0  # Fraction of samples to visualize
    max_realizations_to_display: 6  # Maximum realizations to show
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from multi_sources.data_processing.grid_functions import crop_nan_border_numpy
from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric
from multi_sources.eval.utils import format_tdelta


class VisualEvaluationComparison(AbstractMultisourceEvaluationMetric):
    """Evaluation class that creates visualization figures for targets and predictions."""

    def __init__(
        self,
        model_data,
        parent_results_dir,
        eval_fraction=1.0,
        max_realizations_to_display=3,
        source_name_replacements=None,
        cmap="viridis",
    ):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications
            parent_results_dir (str or Path): Parent directory for all results
            eval_fraction (float): Fraction of samples to visualize (0.0 to 1.0)
            max_realizations_to_display (int): Maximum number of realizations to display
            source_name_replacements (List of tuple of str, optional): List of (pattern, replacement)
                substitutions to apply to source names for display purposes. The replacement
                is done using the re.sub function.
            cmap (str): Colormap to use for visualization.
        """
        super().__init__(
            id_name="visual",
            full_name="Visual Evaluation Comparison",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            source_name_replacements=source_name_replacements,
        )
        self.eval_fraction = eval_fraction
        self.max_realizations_to_display = max_realizations_to_display
        self.cmap = cmap

    def evaluate(self, **kwargs):
        """
        Creates visualization figures for all samples across all models.

        Args:
            **kwargs: Additional keyword arguments (num_workers not used for visualization)
        """
        n_samples = int(self.eval_fraction * self.n_samples)
        for i, (sample_df, sample_data) in enumerate(
            tqdm(self.samples_iterator(), desc="Evaluating samples", total=n_samples)
        ):
            sample_index = sample_df["sample_index"].iloc[0]
            # Choose a channel to plot for each source
            plot_channels = {
                (src_name, src_index): list(
                    sample_data["targets"][(src_name, src_index)].data_vars.keys()
                )[0]
                for src_name, src_index in sample_df.index
            }

            # Crop the padded borders
            cropped_data = self.crop_padded_borders(sample_data, sample_df, plot_channels)

            # Plot the data
            self.plot_sample(
                sample_index,
                cropped_data,
                plot_channels,
                sample_df,
            )

            if i + 1 >= n_samples:
                break

    def crop_padded_borders(self, sample_data, sample_df, plot_channels):
        """Crops the padded borders in the sample data.

        Args:
            sample_data (dict): Dictionary containing targets and predictions for each model
            sample_df (pandas.DataFrame): DataFrame with sample metadata, indexed
                        by (source_name, source_index).
            plot_channels (dict): Dict (src_name, src_index) -> channel to plot
        Returns:
            available_sources (dict): Dict (src_name, src_index) -> available source data
            target_sources (dict): Dict (src_name, src_index) -> target source data
            lats (dict): Dict (src_name, src_index) -> latitude coordinates
            lons (dict): Dict (src_name, src_index) -> longitude coordinates
            predictions (dict): Dict model_id -> (src_name, src_index) -> prediction data
        """

        # Because of the batching process in the prediction pipeline, the data generally
        # includes large padded borders, that we want to crop. In the targets / available sources,
        # and coordinates, these are padded with NaNs. In the predictions,
        # they may contain anything.
        # We will crop them using crop_nan_border_numpy, using the targets as a reference. To do
        # so, we'll here retrieve the targets / available sources, coordinates and predictions
        # for all models, and then crop them all at once.
        available_sources = {}  # (src_name, src_index) -> available / source data
        target_sources = {}  # (src_name, src_index) -> target data
        lats, lons = {}, {}  # (src_name, src_index) -> coordinates data
        predictions = {model_id: {} for model_id in self.model_data}
        for src, target_data in sample_data["targets"].items():
            # Get the availability flag (-1: missing, 0: target, 1: available)
            avail = sample_df.loc[src, "avail"]
            if avail == -1:
                continue
            # Get the coordinates data: latitude and longitude (will be used for the ticklabels)
            src_lat = target_data["lat"].values
            src_lon = target_data["lon"].values
            # Get the channel data
            plot_channel = plot_channels[src]
            channel_data = target_data[plot_channel].values
            # Gather all models' predictions for that source
            preds = [
                sample_data["predictions"][model_id][src][plot_channel].values
                for model_id in self.model_data
            ]
            # Crop all borders all at once using the target as reference
            out = crop_nan_border_numpy(channel_data, [channel_data, src_lat, src_lon] + preds)
            lats[src] = out[1]
            lons[src] = out[2]
            # Store the channel data either as target or available source
            if avail == 0:
                target_sources[src] = out[0]
            else:
                available_sources[src] = out[0]
            # Store the cropped predictions
            for k, model_id in enumerate(self.model_data):
                predictions[model_id][src] = out[k + 3]

        return available_sources, target_sources, lats, lons, predictions

    def plot_sample(self, sample_index, cropped_data, plot_channels, sample_df):
        """Plots the targets and predictions for a single sample.

        Args:
            sample_index (int): Index of the sample
            cropped_data (tuple): Tuple containing available sources, target sources,
                latitudes, longitudes, and predictions
            plot_channels (dict): Dict (src_name, src_index) -> channel to plot
            sample_df (pandas.DataFrame): DataFrame with sample metadata
        """
        available_sources, target_sources, lats, lons, predictions = cropped_data

        # Create a figure with subplots
        num_sources, num_models = len(sample_df), len(self.model_data)
        n_cols = max(num_sources, self.max_realizations_to_display)
        fig, axes = plt.subplots(
            nrows=num_models + 1,
            ncols=n_cols,
            figsize=(3 * n_cols, 3 * (num_models + 1)),
            squeeze=False,
        )

        # ------- FIRST ROW: Targets, then available sources -------
        col_cnt = 0
        # Map {(src_name, src_index) --> mpl Normalize}
        norms = {}
        # First, targets
        for src_name, src_index in target_sources.keys():
            display_name = self._display_src_name(src_name)
            channel_data = target_sources[(src_name, src_index)]
            # Extract the min and max values to create a colormap
            vmin, vmax = np.nanmin(channel_data), np.nanmax(channel_data)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            norms[(src_name, src_index)] = norm

            ax = axes[0, col_cnt]
            ax.imshow(channel_data, aspect="auto", cmap=self.cmap, norm=norm)
            dt = format_tdelta(sample_df.loc[(src_name, src_index), "dt"])
            ax.set_title(f"Target: {display_name} $\delta t=${dt}")
            self._set_coords_as_ticks(ax, lats[(src_name, src_index)], lons[(src_name, src_index)])
            col_cnt += 1

        # Available sources
        for src_name, src_index in available_sources.keys():
            display_name = self._display_src_name(src_name)
            channel_data = available_sources[(src_name, src_index)]
            ax = axes[0, col_cnt]
            ax.imshow(channel_data, aspect="auto", cmap=self.cmap)
            dt = format_tdelta(sample_df.loc[(src_name, src_index), "dt"])
            ax.set_title(f"{display_name} $\delta t=${dt}")
            self._set_coords_as_ticks(ax, lats[(src_name, src_index)], lons[(src_name, src_index)])
            col_cnt += 1

        # Hide the axes that are not used
        for j in range(col_cnt, n_cols):
            axes[0, j].axis("off")

        # ------- SUBSEQUENT ROWS: Model predictions / realizations -----
        for k, model_id in enumerate(self.model_data):
            col_cnt = 0
            for src_name, src_index in target_sources.keys():
                display_name = self._display_src_name(src_name)
                # If the prediction contains one more dim than the target, we
                # assume the first dim is the realization index. In this case,
                # we'll plot one realization per column, up to the maximum number
                # of columns. If there are the same number of dims, there's only
                # a single prediction to plot.
                pred_data = predictions[model_id][(src_name, src_index)]
                if len(pred_data.shape) == len(target_sources[(src_name, src_index)].shape) + 1:
                    # Multiple realizations, plot only the first ones
                    for realization_idx in range(
                        min(self.max_realizations_to_display, pred_data.shape[0])
                    ):
                        ax = axes[k + 1, col_cnt]
                        pred_realization = pred_data[realization_idx, ...]
                        ax.imshow(
                            pred_realization,
                            aspect="auto",
                            cmap=self.cmap,
                            norm=norms[(src_name, src_index)],
                        )
                        dt = format_tdelta(sample_df.loc[(src_name, src_index), "dt"])
                        ax.set_title(f"{display_name} - {model_id} $\delta t=${dt}")
                        self._set_coords_as_ticks(
                            ax, lats[(src_name, src_index)], lons[(src_name, src_index)]
                        )
                        col_cnt += 1
                else:
                    # Single prediction, plot it directly
                    ax = axes[k + 1, col_cnt]
                    ax.imshow(
                        pred_data, aspect="auto", cmap=self.cmap, norm=norms[(src_name, src_index)]
                    )
                    dt = format_tdelta(sample_df.loc[(src_name, src_index), "dt"])
                    ax.set_title(f"{display_name} - {model_id} $\delta t=${dt}")
                    self._set_coords_as_ticks(
                        ax, lats[(src_name, src_index)], lons[(src_name, src_index)]
                    )
                    col_cnt += 1
            # Hide the axes that are not used
            for j in range(col_cnt, n_cols):
                axes[k + 1, j].axis("off")

        plt.tight_layout()
        # Save the figure
        fig_path = self.metric_results_dir / f"sample_{sample_index:04d}.png"
        fig.savefig(fig_path)
        plt.close(fig)

    @staticmethod
    def _set_coords_as_ticks(ax, lats, lons, every_nth_pixel=30):
        """Sets the latitude and longitude coordinates as ticks on the axes."""
        ax.set_xticks(range(0, len(lons[0]), every_nth_pixel))
        ax.set_xticklabels([f"{lon:.2f}" for lon in lons[0, ::every_nth_pixel]], rotation=45)
        ax.set_yticks(range(0, len(lats[:, 0]), every_nth_pixel))
        ax.set_yticklabels([f"{lat:.2f}" for lat in lats[::every_nth_pixel, 0]])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
