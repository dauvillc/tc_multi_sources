"""
Implements the VisualEvaluation class, which just displays the targets and predictions
for a given source.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multi_sources.eval.abstract_evaluation_metric import AbstractEvaluationMetric


class VisualEvaluation(AbstractEvaluationMetric):
    """Displays the targets and predictions for a given source. For each element in each
    batch, the targets and predictions are displayed side by side. The time delta of the
    element is also displayed in the title of the plot.
    """

    def __init__(self, predictions_dir, results_dir):
        super().__init__(
            "visual_eval", "Visualization of predictions", predictions_dir, results_dir
        )

    def evaluate_source(self, source_name, info_df, verbose=True):
        """
        Args:
            source_name (str): Name of the source.
            info_df (pd.DataFrame): DataFrame with at least the following columns:
                batch_idx, index_in_batch, dt.
            verbose (bool): If True, print a progress bar.
        """
        # Create the directory for the source
        source_dir = self.results_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        # Browse the batch indices in the DataFrame
        unique_batch_indices = info_df["batch_idx"].unique()
        iterator = tqdm(unique_batch_indices) if verbose else unique_batch_indices
        for batch_idx in iterator:
            targets, predictions = self.load_batch(source_name, batch_idx)
            # Browse the elements in the batch
            batch_df = info_df[info_df["batch_idx"] == batch_idx]
            for index, row in batch_df.iterrows():
                # Plot the element and save the figure
                index_in_batch = row["index_in_batch"]
                dt = row["dt"]
                target = targets[index_in_batch]
                pred = predictions[index_in_batch] if predictions is not None else None
                fig = self.plot_element(source_name, target, pred, dt)
                fig.savefig(source_dir / f"{batch_idx}_{index_in_batch}.png")
                plt.close(fig)

    def plot_element(self, source_name, target, pred, dt):
        """Plots the target and prediction for a single element.

        Args:
            source_name (str): Name of the source.
            target (np.ndarray): Array of shape (2 + c, height, width),
                where c is the number of channels and the first two
                channels are the latitude and longitude at each pixel.
            pred (np.ndarray or None): Array of shape (channels, height, width),
                or None if no prediction was made.
            dt (float): Time delta of the element.
        """
        # Plot the target on the left and the prediction on the right.
        # Plot the channels from top to bottom. Don't add axes titles.
        num_channels = target.shape[0] - 2
        nrows, ncols = num_channels, 2
        figsize = (8, 2 * num_channels)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        for i in range(num_channels):
            target_values = target[i + 2]  # Don't plot the lat/lon channels
            axes[i, 0].imshow(target_values, cmap="viridis")
            # Set the latitude and longitude as ticklabels on the x and y axes.
            # Use 10 ticks.
            lat, lon = target[:2]
            lat_ticks = np.linspace(lat.min(), lat.max(), num=10)
            lon_ticks = np.linspace(lon.min(), lon.max(), num=10)
            axes[i, 0].set_yticks(np.linspace(0, lat.shape[0], num=10))
            axes[i, 0].set_yticklabels(np.round(lat_ticks, 2))
            axes[i, 0].set_xticks(np.linspace(0, lat.shape[1], num=10))
            axes[i, 0].set_xticklabels(np.round(lon_ticks, 2))
            if pred is not None:
                axes[i, 1].imshow(pred[i], cmap="viridis")
            # Use the same ticks as for the target
            axes[i, 1].set_yticks(np.linspace(0, lat.shape[0], num=10))
            axes[i, 1].set_yticklabels(np.round(lat_ticks, 2))
            axes[i, 1].set_xticks(np.linspace(0, lat.shape[1], num=10))
            axes[i, 1].set_xticklabels(np.round(lon_ticks, 2))
        # Indicate "Target" on the left and "Prediction" on the right
        axes[0, 0].set_title("Target")
        axes[0, 1].set_title("Prediction")
        # Add the source name and the time delta to the title
        title = f"{source_name} - dt = {dt}h"
        fig.suptitle(title)
        return fig
