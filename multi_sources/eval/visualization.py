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

    def __init__(self, model_data, parent_results_dir, eval_fraction=1.0):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to dictionaries containing:
                - info_df: DataFrame with metadata
                - root_dir: Path to predictions directory
                - results_dir: Path to results directory
                - run_id: Run ID
                - pred_name: Prediction name
            parent_results_dir (Path): Parent directory for all results
            eval_fraction (float): Fraction of the dataset to evaluate. If less than
                1.0, a random subset of the dataset will be displayed.
        """
        super().__init__(
            "visual_eval", "Visualization of predictions", model_data, parent_results_dir
        )
        self.eval_fraction = eval_fraction

    def evaluate_sources(self, verbose=True, num_workers=0):
        """
        Args:
            verbose (bool): Whether to show progress bars.
            num_workers (int): Number of workers for parallel processing.
        """
        sns.set_context("paper")

        # Process each model separately
        for model_id, data in self.model_data.items():
            run_id = data["run_id"]
            pred_name = data["pred_name"]
            info_df = data["info_df"]

            print(
                f"\nCreating visualizations for model: {model_id} (run_id: {run_id}, prediction: {pred_name})"
            )

            # Get the results directory for this model
            model_results_dir = self.model_dirs[model_id]["results_dir"]

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
                            src, index, batch_idx, model_id
                        )

                    # Display the targets and predictions
                    display_batch(
                        batch_info, batch_idx, targets, preds, model_results_dir, model_id
                    )
            else:
                # Split batches into chunks for parallel processing
                batch_chunks = np.array_split(unique_batch_indices, num_workers)
                print(
                    f"Dividing {len(unique_batch_indices)} batches into {len(batch_chunks)} chunks"
                )
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
                                    model_results_dir,
                                    verbose,
                                    i,  # Pass the process ID (chunk index)
                                    model_id,  # Pass the model ID
                                )
                            )

                    for future in tqdm(futures, desc="Processing chunks", disable=not verbose):
                        future.result()


def display_batch(batch_info, batch_idx, targets, preds, results_dir, model_id):
    """Displays the targets and predictions for a given batch index. Auxiliary function
    for parallel execution in the VisualEvaluation class.
    Args:
        batch_info (pd.DataFrame): DataFrame with the information of the batch.
        batch_idx (int): Index of the batch to display.
        targets (dict): Dictionary with the targets for each source/index pair.
        preds (dict): Dictionary with the predictions for each source/index pair.
        results_dir (Path): Directory where the results will be saved.
        model_id (str): ID of the model
    """
    # For each sample, create a figure with the targets and predictions
    batch_indices = batch_info["index_in_batch"].unique()
    try:
        for idx in batch_indices:
            sample_df = batch_info[batch_info["index_in_batch"] == idx]
            # For each source, select only the targets and predictions for the sample
            sample_targets, sample_preds = {}, {}
            for _, row in sample_df.iterrows():
                src, index = row["source_name"], row["index"]
                source_pair = (src, index)
                if source_pair in targets and source_pair in preds:
                    if targets[source_pair] is not None:
                        sample_targets[source_pair] = targets[source_pair].isel(samples=idx)
                    if preds[source_pair] is not None:
                        sample_preds[source_pair] = preds[source_pair].isel(samples=idx)

            plot_sample(
                sample_df, sample_targets, sample_preds, results_dir, batch_idx, idx, model_id
            )
    finally:
        # Make sure to close all figures even if an error occurs
        plt.close("all")


def plot_sample(sample_df, targets, preds, results_dir, batch_idx, sample_idx, model_id):
    """Displays in a figure with multiple rows - one row per channel. From left to right:
    available sources, masked source target, and prediction.

    Args:
        sample_df (pd.DataFrame): DataFrame with information about the sample.
        targets (dict): Dictionary with targets for each source/index pair.
        preds (dict): Dictionary with predictions for each source/index pair.
        results_dir (Path): Directory where results will be saved.
        batch_idx (int): Batch index.
        sample_idx (int): Sample index within the batch.
        model_id (str): ID of the model
    """
    # Retrieve the list of all sources/index pairs that are available and not masked
    avail_pairs = sample_df[sample_df["avail"] == 1]

    # Retrieve the source that is masked
    masked_pairs = sample_df[sample_df["avail"] == 0]
    if len(masked_pairs) == 0:
        return  # Skip if no masked source is found

    masked_pair = masked_pairs.iloc[0]
    masked_source, masked_idx = masked_pair["source_name"], masked_pair["index"]

    # Process all sources (both available and masked)
    for _, row in sample_df.iterrows():
        source, idx = row["source_name"], row["index"]
        source_pair = (source, idx)

        # Apply post-processing to all sources based on their type
        if source_pair in targets and targets[source_pair] is not None:
            if "pmw" in source:
                # Process PMW sources
                targets[source_pair], _ = process_pmw(targets[source_pair], None)
            elif source == "tc_primed_era5":
                # Process ERA5 sources
                targets[source_pair], _ = process_era5(targets[source_pair], None)

        # Also process the predictions (only for masked source)
        if source_pair in preds and preds[source_pair] is not None and source == masked_source:
            if "pmw" in source:
                # Process PMW predictions
                _, preds[source_pair] = process_pmw(None, preds[source_pair])
            elif source == "tc_primed_era5":
                # Process ERA5 predictions
                _, preds[source_pair] = process_era5(None, preds[source_pair])

    # Sort pairs by decreasing dt
    avail_pairs.sort_values(by="dt", ascending=False, inplace=True)

    # Get all unique channels across all sources
    all_channels = set()
    for _, row in avail_pairs.iterrows():
        source = row["source_name"]
        source_pair = (source, row["index"])
        if source_pair in targets and targets[source_pair] is not None:
            target_ds = targets[source_pair]
            all_channels.update(target_ds.data_vars)

    # Add channels from masked source
    masked_pair_key = (masked_source, masked_idx)
    if masked_pair_key in targets and targets[masked_pair_key] is not None:
        masked_target_ds = targets[masked_pair_key]
        all_channels.update(masked_target_ds.data_vars)
    all_channels = sorted(list(all_channels))  # Sort for consistent ordering

    # Calculate number of rows (one per channel) and columns
    num_rows = len(all_channels)
    num_avail_sources = len(avail_pairs)
    # 2 columns per masked channel (target + prediction)
    num_cols = num_avail_sources + 2 + 1  # +1 for the separator

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(3 * num_cols, 2.5 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.01, hspace=0.15)

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

    # For each channel, plot available sources on the corresponding row
    for row_idx, channel in enumerate(all_channels):
        # Plot each available source on this row if it has this channel
        for col_idx, (_, row) in enumerate(avail_pairs.iterrows()):
            source, idx = row["source_name"], row["index"]
            source_pair = (source, idx)

            if source_pair in targets and targets[source_pair] is not None:
                target_ds = targets[source_pair]

                # Check if this source has this channel
                if channel in target_ds.data_vars:
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    target = target_ds[channel].values
                    target = crop_nan_border_numpy(target, [target])[0]

                    ax.imshow(target, cmap="viridis")
                    # Make title more compact
                    displayed_name = " ".join(source.split("_")[3:5])
                    dt_str = strfdelta(row["dt"], "%D d%H:%M")
                    title = f"{displayed_name}\n$\delta_t$={dt_str}"
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                else:
                    # Empty subplot if source doesn't have this channel
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    ax.axis("off")
            else:
                # Empty subplot if source doesn't have this channel
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.axis("off")

        # Add vertical separator line
        ax = fig.add_subplot(gs[row_idx, num_avail_sources])
        ax.axvline(x=0.5, color="black", linestyle="-", linewidth=2)
        ax.axis("off")

        # Get target and prediction for masked source for this channel
        masked_pair_key = (masked_source, masked_idx)

        if masked_pair_key in targets and targets[masked_pair_key] is not None:
            masked_target_ds = targets[masked_pair_key]

            if channel in masked_target_ds.data_vars:
                # Check if the channel exists in predictions
                pred_ds = preds.get(masked_pair_key)
                channel_in_pred = pred_ds is not None and channel in pred_ds.data_vars

                if channel_in_pred:
                    target = masked_target_ds[channel].values
                    pred = pred_ds[channel].values
                    # Crop NaN borders using only the target values
                    target, pred = crop_nan_border_numpy(target, [target, pred])

                    # Calculate shared min/max values for this channel
                    vmin = min(np.nanmin(target), np.nanmin(pred))
                    vmax = max(np.nanmax(target), np.nanmax(pred))

                    # Plot masked source target
                    dt = sample_df[sample_df["source_name"] == masked_source]["dt"].iloc[0]
                    ax = fig.add_subplot(gs[row_idx, num_avail_sources + 1])
                    ax.imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)

                    displayed_name = " ".join(masked_source.split("_")[3:5])
                    dt_str = strfdelta(dt, "%D d%H:%M")
                    title = f"{displayed_name}\n$\delta_t$={dt_str}"
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")

                    # Plot prediction
                    ax = fig.add_subplot(gs[row_idx, num_avail_sources + 2])
                    ax.imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
                    title = f"{displayed_name}\n$\delta_t$={dt_str}"
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                else:
                    # Empty subplot with message if channel not in predictions
                    ax = fig.add_subplot(gs[row_idx, num_avail_sources + 1])
                    ax.axis("off")
                    ax = fig.add_subplot(gs[row_idx, num_avail_sources + 2])
                    ax.axis("off")
                    ax.set_title(f"{displayed_name}\nPrediction N/A", color="gray")
            else:
                # Empty subplot for target and prediction if masked source doesn't have this channel
                ax = fig.add_subplot(gs[row_idx, num_avail_sources + 1])
                ax.axis("off")
                ax = fig.add_subplot(gs[row_idx, num_avail_sources + 2])
                ax.axis("off")
        else:
            # Empty subplot for target and prediction if masked source doesn't have this channel
            ax = fig.add_subplot(gs[row_idx, num_avail_sources + 1])
            ax.axis("off")
            ax = fig.add_subplot(gs[row_idx, num_avail_sources + 2])
            ax.axis("off")

    # Save the figure with tight layout and minimal padding
    plt.tight_layout(
        rect=[0.05, 0, 1, 0.95]
    )  # Adjust layout, leaving minimal space for row labels
    fig.savefig(
        results_dir / f"{batch_idx}_{sample_idx}.png", bbox_inches="tight", pad_inches=0.1, dpi=150
    )
    plt.close(fig)


def process_batch_chunk(
    chunk_df,
    batch_indices,
    load_batch_fn,
    results_dir,
    verbose=True,
    process_id=None,
    model_id=None,
):
    """Process a chunk of batches in a single worker process.

    Args:
        chunk_df (pd.DataFrame): DataFrame containing information for a chunk of batches
        batch_indices (np.array): Array
        load_batch_fn (callable): Function to load batch data (from VisualEvaluation.load_batch)
        results_dir (Path): Directory where results will be saved
        verbose (bool): Whether to display progress information
        process_id (int, optional): ID of the process. If provided, only process 0 will display progress.
        model_id (str, optional): ID of the model
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
            targets[(src, index)], preds[(src, index)] = load_batch_fn(
                src, index, batch_idx, model_id
            )

        # Display the targets and predictions
        display_batch(batch_info, batch_idx, targets, preds, results_dir, model_id)

    # Make sure to close all figures to avoid memory leaks
    plt.close("all")

    return True  # Return value to indicate completion


def apply_post_processing(targets, preds, source_name):
    """Apply post-processing to targets and predictions based on source type.

    Args:
        targets (xarray.Dataset): Target dataset
        preds (xarray.Dataset): Predictions dataset
        source_name (str): Name of the source

    Returns:
        tuple: Processed (targets, preds)
    """
    if source_name == "tc_primed_era5":
        return process_era5(targets, preds)
    elif "pmw" in source_name:
        return process_pmw(targets, preds)
    else:
        # For other sources, return the datasets unchanged
        return targets, preds


def process_era5(targets, preds, radius_around_center_km=1000):
    """Post-process ERA5 data. Keep the u and v wind components."""
    # Return the datasets unchanged, preserving u_wind_10m and v_wind_10m
    return targets, preds


def process_pmw(targets, preds, radius_around_center_km=1000):
    """Post-process PMW data by renaming the variable whose name contains "TB" to "Brightness Temperature".

    Args:
        targets (xarray.Dataset): Target dataset
        preds (xarray.Dataset): Predictions dataset
        radius_around_center_km (int): Radius around center in km for masking (not used in visualization)

    Returns:
        tuple: Processed (targets, preds)
    """
    # Rename the variable whose name contains "TB" to "Brightness Temperature"
    target_result = targets
    pred_result = preds

    if targets is not None:
        tb_var = [var for var in targets.data_vars if "TB" in var]
        if len(tb_var) == 1:
            target_result = targets.rename({tb_var[0]: "Brightness Temperature"})

    if preds is not None:
        tb_var = [var for var in preds.data_vars if "TB" in var]
        if len(tb_var) == 1:
            pred_result = preds.rename({tb_var[0]: "Brightness Temperature"})

    return target_result, pred_result
