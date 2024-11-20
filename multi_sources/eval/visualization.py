"""
Implements the VisualEvaluation class, which just displays the targets and predictions
for a given source.
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
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

    def evaluate_sources(self, info_df, verbose=True, num_workers=0):
        """
        Args:
            info_df (pd.DataFrame): DataFrame with at least the following columns:
                source_name, avail, batch_idx, index_in_batch, dt.
            **kwargs: Additional keyword arguments.
        """
        # Browse the batch indices in the DataFrame
        unique_batch_indices = info_df["batch_idx"].unique()
        if num_workers < 2:
            for batch_idx in tqdm(unique_batch_indices, desc="Batches", disable=not verbose):
                # Get the sources included in the batch
                batch_info = info_df[info_df["batch_idx"] == batch_idx]
                sources = batch_info["source_name"].unique()
                # For each source, load the targets and predictions
                targets, preds = {}, {}
                for source in sources:
                    targets[source], preds[source] = self.load_batch(source, batch_idx)
                # Display the targets and predictions
                display_batch(batch_info, batch_idx, targets, preds, self.results_dir)
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for batch_idx in unique_batch_indices:
                    # Get the sources included in the batch
                    batch_info = info_df[info_df["batch_idx"] == batch_idx]
                    sources = batch_info["source_name"].unique()
                    # For each source, load the targets and predictions
                    targets, preds = {}, {}
                    for source in sources:
                        targets[source], preds[source] = self.load_batch(source, batch_idx)
                    futures.append(
                        executor.submit(
                            display_batch, batch_info, batch_idx, targets, preds, self.results_dir
                        )
                    )
                for future in tqdm(futures, desc="Batches", disable=not verbose):
                    future.result()


def display_batch(batch_info, batch_idx, targets, preds, results_dir):
    """Displays the targets and predictions for a given batch index. Auxiliary function
    for parallel execution in the VisualEvaluation class.
    Args:
        batch_info (pd.DataFrame): DataFrame with the information of the batch.
        batch_idx (int): Index of the batch to display.
        targets (dict): Dictionary with the targets for each source.
        preds (dict): Dictionary with the predictions for each source.
        results_dir (Path): Directory where the results will be saved.
    """
    sources = batch_info["source_name"].unique()
    S = len(sources)
    # For each sample, create a figure with the targets and predictions
    batch_indices = batch_info["index_in_batch"].unique()
    for idx in batch_indices:
        # Create a figure with two columns (target and prediction) and S rows
        fig, axes = plt.subplots(nrows=S, ncols=2, figsize=(10, 5 * S))
        for i, source in enumerate(sources):
            target_ds = targets[source]
            pred_ds = preds[source]
            if target_ds is not None:
                target_sample = target_ds.isel(samples=idx)
                lat = target_sample['lat'].values
                lon = target_sample['lon'].values
                target = target_sample['targets'].values[0]  # Assuming first channel

                axes[i, 0].imshow(target, cmap="viridis")
                axes[i, 0].set_title(f"{source} - target")

                # Set axis ticks and labels
                set_axis_ticks(axes[i, 0], lat, lon)

                sample_info = batch_info[
                    (batch_info["source_name"] == source) & (batch_info["index_in_batch"] == idx)
                ]

                if pred_ds is not None and sample_info["avail"].item() == 0:
                    pred_sample = pred_ds.isel(samples=idx)
                    pred = pred_sample['outputs'].values[0]  # Assuming first channel

                    axes[i, 1].imshow(pred, cmap="viridis")
                    axes[i, 1].set_title(f"pred. - dt={sample_info['dt'].item()}")
                    set_axis_ticks(axes[i, 1], lat, lon)
                else:
                    axes[i, 1].set_title(f"{source} - prediction not available")
            else:
                axes[i, 0].set_title(f"{source} - target not available")
                axes[i, 1].set_title(f"{source} - prediction not available")
        # Save the figure
        plt.tight_layout()
        plt.savefig(results_dir / f"{batch_idx}_{idx}.png")
        plt.close(fig)

def set_axis_ticks(ax, lat, lon):
    """Helper function to set axis ticks and labels."""
    H, W = lat.shape
    lat_ticks = np.linspace(0, H - 1, num=10).astype(int)
    lon_ticks = np.linspace(0, W - 1, num=10).astype(int)
    lat_labels = lat[lat_ticks, 0].round(2)
    lon_labels = lon[0, lon_ticks].round(2)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels(lon_labels)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels(lat_labels)
    ax.tick_params(axis="x", rotation=45)
