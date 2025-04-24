"""
Implements the VisualEvaluation class, which just displays the targets and predictions
for a given source.
"""

from concurrent.futures import ProcessPoolExecutor
from string import Template

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from multi_sources.data_processing.grid_functions import crop_nan_border_numpy
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
    d["H"], d["M"], d["S"] = f"{d['H']:02d}", f"{d['M']:02d}", f"{d['S']:02d}"
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
        sns.set_context("paper")
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
                # For each source, load the targets and predictions
                targets, preds = {}, {}
                for _, row in batch_info.iterrows():
                    src, index = row["source_name"], row["index"]
                    targets[(src, index)], preds[(src, index)] = self.load_batch(
                        src, index, batch_idx
                    )
                # Display the targets and predictions
                display_batch(batch_info, batch_idx, targets, preds, self.results_dir)
        else:
            # Split batches into chunks for parallel processing
            batch_chunks = np.array_split(unique_batch_indices, num_workers)
            print(f"Dividing {len(unique_batch_indices)} batches into {len(batch_chunks)} chunks")
            futures = []

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for i, chunk in enumerate(batch_chunks):
                    # Create chunks of batch indices
                    chunk_df = info_df[info_df["batch_idx"].isin(chunk)]
                    if not chunk_df.empty:
                        futures.append(
                            executor.submit(
                                process_batch_chunk,
                                chunk_df,
                                chunk,
                                self.load_batch,
                                self.results_dir,
                                verbose,
                                i,  # Pass the process ID (chunk index)
                            )
                        )

                for future in tqdm(futures, desc="Processing chunks", disable=not verbose):
                    future.result()


def display_batch(batch_info, batch_idx, targets, preds, results_dir):
    """Displays the targets and predictions for a given batch index. Auxiliary function
    for parallel execution in the VisualEvaluation class.
    Args:
        batch_info (pd.DataFrame): DataFrame with the information of the batch.
        batch_idx (int): Index of the batch to display.
        targets (dict): Dictionary with the targets for each source/index pair.
        preds (dict): Dictionary with the predictions for each source/index pair.
        results_dir (Path): Directory where the results will be saved.
    """
    # For each sample, create a figure with the targets and predictions
    batch_indices = batch_info["index_in_batch"].unique()
    for idx in batch_indices:
        sample_df = batch_info[batch_info["index_in_batch"] == idx]
        # For each source, select only the targets and predictions for the sample
        sample_targets, sample_preds = {}, {}
        for _, row in sample_df.iterrows():
            src, index = row["source_name"], row["index"]
            sample_targets[(src, index)] = targets[(src, index)].isel(samples=idx)
            sample_preds[(src, index)] = preds[(src, index)].isel(samples=idx)
        plot_sample(sample_df, sample_targets, sample_preds, results_dir, batch_idx, idx)


def plot_sample(sample_df, targets, preds, results_dir, batch_idx, sample_idx):
    """Displays in a figure with multiple rows - one row per channel. From left to right:
    available sources, masked source target, and prediction."""
    # Retrieve the list of all sources/index pairs that are available and not masked
    avail_pairs = sample_df[sample_df["avail"] == 1]
    # Retrieve the source that is masked
    masked_pair = sample_df[sample_df["avail"] == 0].iloc[0]
    masked_source, masked_idx = masked_pair["source_name"], masked_pair["index"]

    # Sort pairs by decreasing dt
    avail_pairs.sort_values(by="dt", ascending=False, inplace=True)

    # Get all unique channels across all sources
    all_channels = set()
    for _, row in avail_pairs.iterrows():
        source = row["source_name"]
        target_ds = targets[(source, row["index"])]
        all_channels.update(target_ds.data_vars)

    # Add channels from masked source
    masked_target_ds = targets[(masked_source, masked_idx)]
    all_channels.update(masked_target_ds.data_vars)
    all_channels = sorted(list(all_channels))  # Sort for consistent ordering

    # Calculate number of rows (one per channel) and columns
    num_rows = len(all_channels)
    num_avail_sources = len(avail_pairs)
    # 2 columns per masked channel (target + prediction)
    num_cols = num_avail_sources + 2 + 1  # +1 for the separator

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(3 * num_cols, 2.5 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.4, hspace=0.3)

    # Add channel labels on the left side
    for i, channel in enumerate(all_channels):
        fig.text(
            0.01,  # Left aligned
            1 - (i + 0.5) / num_rows,  # Centered vertically for each row
            channel,
            fontsize=12,
            weight="bold",
            ha="left",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", pad=3),
        )

    # Add section labels at the top
    fig.text(
        (num_avail_sources / 2) / num_cols,
        0.98,
        "Available sources",
        fontsize=14,
        weight="bold",
        ha="center",
    )
    fig.text(
        (num_avail_sources + 1 + 1) / num_cols,
        0.98,
        "Target / Prediction",
        fontsize=14,
        weight="bold",
        ha="center",
    )

    # For each channel, plot available sources on the corresponding row
    for row_idx, channel in enumerate(all_channels):
        # Plot each available source on this row if it has this channel
        for col_idx, (_, row) in enumerate(avail_pairs.iterrows()):
            source, idx = row["source_name"], row["index"]
            target_ds = targets[(source, idx)]

            # Check if this source has this channel
            if channel in target_ds.data_vars:
                ax = fig.add_subplot(gs[row_idx, col_idx])
                target = target_ds[channel].values

                # Crop NaN borders from target based on the coordinates
                lat, lon = target_ds.lat.values, target_ds.lon.values
                target, lat, lon = crop_nan_border_numpy(lat, [target, lat, lon])

                dt = sample_df[sample_df["source_name"] == source]["dt"].iloc[0]

                ax.imshow(target, cmap="viridis")
                displayed_name = " ".join(source.split("_")[3:5])
                title = f"{displayed_name}\n$\delta_t=${strfdelta(dt, '%H:%M:%S')}"
                ax.set_title(title)
                set_axis_ticks(ax, lat, lon)
            else:
                # Empty subplot if source doesn't have this channel
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.axis("off")

        # Add vertical separator line
        ax = fig.add_subplot(gs[row_idx, num_avail_sources])
        ax.axvline(x=0.5, color="black", linestyle="-", linewidth=2)
        ax.axis("off")

        # Get target and prediction for masked source for this channel
        if channel in masked_target_ds.data_vars:
            target = masked_target_ds[channel].values

            # Check if the channel exists in predictions
            pred_ds = preds[(masked_source, masked_idx)]
            channel_in_pred = channel in pred_ds.data_vars

            if channel_in_pred:
                pred = pred_ds[channel].values
                # Crop NaN borders
                lat, lon = masked_target_ds.lat.values, masked_target_ds.lon.values
                target, pred, lat, lon = crop_nan_border_numpy(lat, [target, pred, lat, lon])

                # Calculate shared min/max values for this channel
                vmin = min(np.nanmin(target), np.nanmin(pred))
                vmax = max(np.nanmax(target), np.nanmax(pred))
            else:
                # Only process target if channel not in predictions
                lat, lon = masked_target_ds.lat.values, target_ds.lon.values
                target, lat, lon = crop_nan_border_numpy(lat, [target, lat, lon])

                # Calculate min/max values for target only
                vmin = np.nanmin(target)
                vmax = np.nanmax(target)

            # Plot masked source target
            dt = sample_df[sample_df["source_name"] == masked_source]["dt"].iloc[0]
            ax = fig.add_subplot(gs[row_idx, num_avail_sources + 1])
            ax.imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)

            displayed_name = " ".join(masked_source.split("_")[3:5])
            title = f"{displayed_name}\n$\delta_t=${strfdelta(dt, '%H:%M:%S')}\n(MASKED)"
            ax.set_title(title)
            set_axis_ticks(ax, lat, lon)

            # Plot prediction
            ax = fig.add_subplot(gs[row_idx, num_avail_sources + 2])
            if channel_in_pred:
                ax.imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
                title = f"{displayed_name}\n$\delta_t=${strfdelta(dt, '%H:%M:%S')}\nPrediction"
                ax.set_title(title)
                set_axis_ticks(ax, lat, lon)
            else:
                # Empty subplot with message if channel not in predictions
                ax.axis("off")
                ax.set_title(f"{displayed_name}\nPrediction N/A", color="gray")
        else:
            # Empty subplot for target and prediction if masked source doesn't have this channel
            ax = fig.add_subplot(gs[row_idx, num_avail_sources + 1])
            ax.axis("off")
            ax = fig.add_subplot(gs[row_idx, num_avail_sources + 2])
            ax.axis("off")

    # Save the figure
    plt.tight_layout(rect=[0.05, 0, 1, 0.98])  # Adjust layout, leaving space for row labels
    fig.savefig(results_dir / f"{batch_idx}_{sample_idx}.png", bbox_inches="tight")
    plt.close(fig)


def process_batch_chunk(
    chunk_df, batch_indices, load_batch_fn, results_dir, verbose=True, process_id=None
):
    """Process a chunk of batches in a single worker process.

    Args:
        chunk_df (pd.DataFrame): DataFrame containing information for a chunk of batches
        batch_indices (np.array): Array of batch indices to process
        load_batch_fn (callable): Function to load batch data (from VisualEvaluation.load_batch)
        results_dir (Path): Directory where results will be saved
        verbose (bool): Whether to display progress information
        process_id (int, optional): ID of the process. If provided, only process 0 will display progress.
    """
    # Only show progress for process_id=0 or if process_id is None
    show_progress = verbose and (process_id is None or process_id == 0)

    for batch_idx in tqdm(
        batch_indices,
        desc=f"Batches in chunk {process_id if process_id is not None else ''}",
        disable=not show_progress,
        leave=False,
    ):
        # Get the sources included in the batch
        batch_info = chunk_df[chunk_df["batch_idx"] == batch_idx]

        # For each source, load the targets and predictions
        targets, preds = {}, {}
        for _, row in batch_info.iterrows():
            src, index = row["source_name"], row["index"]
            targets[(src, index)], preds[(src, index)] = load_batch_fn(src, index, batch_idx)

        # Display the targets and predictions
        display_batch(batch_info, batch_idx, targets, preds, results_dir)

    return True  # Return value to indicate completion


def set_axis_ticks(ax, lat, lon):
    """Helper function to set axis ticks and labels."""
    H, W = lat.shape
    lat_ticks = np.linspace(0, H - 1, num=10).astype(int)
    lon_ticks = np.linspace(0, W - 1, num=10).astype(int)
    lat_labels = np.nanmean(lat[lat_ticks], axis=1).round(2)
    lon_labels = np.nanmean(lon[:, lon_ticks], axis=0).round(2)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels(lon_labels)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels(lat_labels)
    ax.tick_params(axis="x", rotation=45)
