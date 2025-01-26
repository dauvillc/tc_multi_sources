"""
Implements the VisualEvaluation class, which just displays the targets and predictions
for a given source.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import Template
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


# A little code snippet from Shawn Chin & Peter Mortensen
# https://stackoverflow.com/questions/8906926/formatting-timedelta-objects/8907269#8907269
class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    """equivalent of strftime for timedelta objects"""
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


class VisualEvaluation(AbstractMultisourceEvaluationMetric):
    """Displays the targets and predictions for a given source. For each sample in each
    batch:
    - Retrieves the list of sources that were included in the batch.
    - Loads the targets and predictions for the source.
    - Creates a figure with two columns (target and prediction) S rows (one per source in
        the batch).
    - Saves the figure to the results directory.
    """

    def __init__(self, predictions_dir, results_dir, eval_fraction=1.0):
        """
        Args:
            predictions_dir (Path): Directory with the predictions.
            results_dir (Path): Directory where the results will be saved.
            eval_fraction (float): Fraction of the dataset to evaluate. If less than
                1.0, a random portion of the dataset will be displayed.
        """
        super().__init__(
            "visual_eval", "Visualization of predictions", predictions_dir, results_dir
        )
        self.eval_fraction = eval_fraction

    def evaluate_sources(self, info_df, verbose=True, num_workers=0):
        """
        Args:
            info_df (pd.DataFrame): DataFrame with at least the following columns:
                source_name, avail, batch_idx, index_in_batch, dt.
            **kwargs: Additional keyword arguments.
        """
        # Browse the batch indices in the DataFrame
        unique_batch_indices = info_df["batch_idx"].unique()
        # If eval_fraction < 1.0, select a random subset of the batch indices
        if self.eval_fraction < 1.0:
            num_batches = len(unique_batch_indices)
            num_eval_batches = np.ceil(self.eval_fraction * num_batches).astype(int)
            unique_batch_indices = np.random.choice(
                unique_batch_indices, num_eval_batches, replace=False
            )
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
        fig.suptitle(
            f"Left: Targets - Right: Predictions - batch_idx={batch_idx}, idx={idx}", y=1.05
        )
        for i, source in enumerate(sources):
            target_ds = targets[source]
            pred_ds = preds[source]
            if target_ds is not None:
                sample_info = batch_info[
                    (batch_info["source_name"] == source) & (batch_info["index_in_batch"] == idx)
                ]
                dt = pd.to_timedelta(sample_info["dt"].values[0])
                axes[i, 0].set_title(f"{source} - dt={strfdelta(dt, '%D d%H h:%M min')}")
                # Check the spatial shape of the sample. If it is an image (2d shape),
                # display it.
                all_channels = list(target_ds.data_vars.keys())
                sample_shape = sample_info["spatial_shape"].values[0]
                if len(sample_shape) == 2:
                    channel = all_channels[1]
                    target_sample = target_ds.isel(samples=idx)
                    lat = target_sample["lat"].values
                    lon = target_sample["lon"].values
                    target = target_sample[channel].values  # Display the first channel
                    axes[i, 0].imshow(target, cmap="viridis")

                    # Set axis ticks and labels
                    set_axis_ticks(axes[i, 0], lat, lon)

                    # The model only made a prediction when the sample was masked,
                    # which corresponds to avail=0 (-1 for unavailable and 1 for available
                    # but not masked).
                    if pred_ds is not None and sample_info["avail"].item() == 0:
                        pred_sample = pred_ds.isel(samples=idx)
                        pred = pred_sample[channel].values  # first channel

                        axes[i, 1].imshow(pred, cmap="viridis")
                        set_axis_ticks(axes[i, 1], lat, lon)
                    else:
                        axes[i, 1].set_title(f"Prediction not available")
                else:
                    # For scalar data, just display the target and prediction values
                    # as text.
                    target = target_ds.isel(samples=idx)
                    for j, channel in enumerate(all_channels):
                        value = target[channel].values.item()
                        axes[i, 0].text(0.5, 0.5 - 0.1 * j, f"{channel}: {value}", ha="center")
                    # Display the prediction if available
                    if pred_ds is not None and sample_info["avail"].item() == 0:
                        axes[i, 1].set_title(f"pred. - dt={strfdelta(dt, '%D days %H:%M')}")
                        pred = pred_ds.isel(samples=idx)
                        for j, channel in enumerate(all_channels):
                            value = pred[channel].values.item()
                            axes[i, 1].text(
                                0.5, 0.5 - 0.1 * j, f"{channel}: {value}", ha="center"
                            )
            else:
                axes[i, 0].set_title(f"{source} not available")
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
