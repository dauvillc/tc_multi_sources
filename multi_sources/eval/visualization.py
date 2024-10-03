"""
Implements the VisualEvaluation class, which just displays the targets and predictions
for a given source.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


class VisualEvaluation(AbstractMultisourceEvaluationMetric):
    """Displays the targets and predictions for a given source. For each sample in each
    batch:
    - Retrieves the list of sources that were included in the batch.
    - Loads the targets and predictions for the source.
    - Creates a figure with two columns (target and prediction) S rows (one per source in
        the batch).
    - Saves the figure to the results directory.
    """

    def __init__(self, predictions_dir, results_dir):
        super().__init__(
            "visual_eval", "Visualization of predictions", predictions_dir, results_dir
        )

    def evaluate_sources(self, info_df, verbose=True):
        """
        Args:
            info_df (pd.DataFrame): DataFrame with at least the following columns:
                source_name, batch_idx, index_in_batch, dt.
            **kwargs: Additional keyword arguments.
        """
        # Browse the batch indices in the DataFrame
        unique_batch_indices = info_df["batch_idx"].unique()
        iterator = tqdm(unique_batch_indices) if verbose else unique_batch_indices
        for batch_idx in iterator:
            # Get the sources included in the batch
            batch_info = info_df[info_df["batch_idx"] == batch_idx]
            sources = batch_info["source_name"].unique()
            S = len(sources)
            # For each source, load the targets and predictions
            targets, preds = {}, {}
            for source in sources:
                targets[source], preds[source] = self.load_batch(source, batch_idx)
            # For each sample, create a figure with the targets and predictions
            batch_indices = batch_info["index_in_batch"].unique()
            for idx in batch_indices:
                # Create a figure with two columns (target and prediction) and S rows
                fig, axes = plt.subplots(nrows=S, ncols=2, figsize=(10, 5 * S))
                for i, source in enumerate(sources):
                    target = targets[source][idx]
                    pred = preds[source][idx]
                    # The target has shape (2 + C, H, W), where the first two channels are
                    # the latitude and longitude. We'll use those for the Y and X axes ticks,
                    # respectively.
                    # We'll only show the first channel of the values.
                    lat, lon, target = target[0], target[1], target[2:][0]
                    # Show the target on the left.
                    axes[i, 0].imshow(target, cmap="viridis")
                    axes[i, 0].set_title(f"{source} - target")
                    # Set 10 ticks on each axis, and use the lat/lon values as labels
                    lat_ticks = np.linspace(0, target.shape[0] - 1, num=10).astype(int)
                    lon_ticks = np.linspace(0, target.shape[1] - 1, num=10).astype(int)
                    lon_labels = lon[0][lon_ticks].round(2)
                    lat_labels = lat[:, 0][lat_ticks].round(2)
                    axes[i, 0].set_xticks(lon_ticks)
                    axes[i, 0].set_xticklabels(lon_labels)
                    axes[i, 0].set_yticks(lat_ticks)
                    axes[i, 0].set_yticklabels(lat_labels)
                    # For the longitude labels, use a 45-degree rotation
                    axes[i, 0].tick_params(axis="x", rotation=45)
                    # Show the prediction on the right, if the prediction is not None
                    if pred is not None:
                        axes[i, 1].imshow(pred[0], cmap="viridis")
                        # Indicate the dt in the title
                        sample_info = batch_info[(batch_info["source_name"] == source)
                                                 & (batch_info["index_in_batch"] == idx)]
                        dt = sample_info["dt"].values[0]
                        axes[i, 1].set_title(f"pred. - dt={dt:.2f}h")
                        # Set the same ticks as for the target
                        axes[i, 1].set_xticks(lon_ticks)
                        axes[i, 1].set_xticklabels(lon_labels)
                        axes[i, 1].set_yticks(lat_ticks)
                        axes[i, 1].set_yticklabels(lat_labels)
                        axes[i, 1].tick_params(axis="x", rotation=45)
                # Save the figure
                plt.tight_layout()
                plt.savefig(self.results_dir / f"{batch_idx}_{idx}.png")
                plt.close(fig)