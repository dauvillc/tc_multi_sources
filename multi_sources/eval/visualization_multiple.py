"""
Implements the VisualEvaluationMultiple class, which displays the targets and predictions
for a given source when the predictions contain multiple realizations.
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


class VisualEvaluationMultiple(AbstractMultisourceEvaluationMetric):
    """Displays the targets and all realizations of predictions for a given source. For each sample in each
    batch:
    - Retrieves the list of sources that were included in the batch.
    - Loads the targets and predictions for the source.
    - Creates a figure with multiple columns (target and all realizations) and S rows (one per source in
        the batch).
    - Saves the figure to the results directory.
    """

    def __init__(
        self,
        model_data,
        parent_results_dir,
        eval_fraction=1.0,
        max_realizations_to_display=6,
        include_pred_mean=True,
    ):
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
            max_realizations_to_display (int): Maximum number of realizations to display in the figure.
                If the predictions contain more realizations, only the first `max_realizations_to_display`
                will be shown.
            include_pred_mean (bool): If True, expects the first realization in preds to be
                the predicted mean.
                It is then displayed as the first column after the target.
        """
        super().__init__(
            "visual_eval_multiple",
            "Visualization of multi-realization predictions",
            model_data,
            parent_results_dir,
        )
        self.eval_fraction = eval_fraction
        self.max_realizations_to_display = max_realizations_to_display
        self.include_pred_mean = include_pred_mean

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
                f"\nCreating multi-realization visualizations for model: {model_id} (run_id: {run_id}, prediction: {pred_name})"
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
                    display_batch_multiple(
                        batch_info,
                        batch_idx,
                        targets,
                        preds,
                        model_results_dir,
                        model_id,
                        self.max_realizations_to_display,
                        self.include_pred_mean,
                    )
            else:
                # Use parallel processing
                chunk_size = max(1, len(unique_batch_indices) // num_workers)
                chunks = [
                    unique_batch_indices[i : i + chunk_size]
                    for i in range(0, len(unique_batch_indices), chunk_size)
                ]

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for i, chunk in enumerate(chunks):
                        chunk_df = info_df[info_df["batch_idx"].isin(chunk)]
                        future = executor.submit(
                            process_batch_chunk_multiple,
                            chunk_df,
                            chunk,
                            self.load_batch,
                            model_results_dir,
                            verbose and i == 0,  # Only show progress for the first worker
                            i,
                            model_id,
                            self.max_realizations_to_display,
                            self.include_pred_mean,
                        )
                        futures.append(future)

                    # Wait for all workers to complete
                    for future in futures:
                        future.result()


def display_batch_multiple(
    batch_info,
    batch_idx,
    targets,
    preds,
    results_dir,
    model_id,
    max_realizations_to_display,
    include_pred_mean=True,
):
    """Displays the targets and all realizations of predictions for a given batch index. Auxiliary function
    for parallel execution in the VisualEvaluationMultiple class.
    Args:
        batch_info (pd.DataFrame): DataFrame with the information of the batch.
        batch_idx (int): Index of the batch to display.
        targets (dict): Dictionary with the targets for each source/index pair.
        preds (dict): Dictionary with the predictions for each source/index pair.
        results_dir (Path): Directory where the results will be saved.
        model_id (str): ID of the model
        max_realizations_to_display (int): Maximum number of realizations to display.
        include_pred_mean (bool): If True, expects the first realization in preds to be the predicted mean.
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

            plot_sample_multiple(
                sample_df,
                sample_targets,
                sample_preds,
                results_dir,
                batch_idx,
                idx,
                model_id,
                max_realizations_to_display,
                include_pred_mean=include_pred_mean,
            )
    finally:
        # Make sure to close all figures even if an error occurs
        plt.close("all")


def plot_sample_multiple(
    sample_df,
    targets,
    preds,
    results_dir,
    batch_idx,
    sample_idx,
    model_id,
    max_realizations_to_display,
    include_pred_mean=True,
):
    """Displays in a figure with multiple rows - one row per channel. From left to right:
    available sources, masked source target, and all realizations of predictions.

    Args:
        sample_df (pd.DataFrame): DataFrame with information about the sample.
        targets (dict): Dictionary with targets for each source/index pair.
        preds (dict): Dictionary with predictions for each source/index pair.
        results_dir (Path): Directory where results will be saved.
        batch_idx (int): Batch index.
        sample_idx (int): Sample index within the batch.
        model_id (str): ID of the model
        max_realizations_to_display (int): Maximum number of realizations to display.
        include_pred_mean (bool): If True, expects the first realization in preds to be the predicted mean.
            It is then displayed as the first column after the target.
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
        source_name, index = row["source_name"], row["index"]
        source_pair = (source_name, index)
        if source_pair in targets and targets[source_pair] is not None:
            # Apply post-processing to the target/prediction datasets
            targets[source_pair], preds[source_pair] = apply_post_processing(
                targets[source_pair], preds.get(source_pair), source_name
            )

    # Sort pairs by decreasing dt
    avail_pairs.sort_values(by="dt", ascending=False, inplace=True)

    # Get all unique channels across all sources
    all_channels = set()
    for _, row in avail_pairs.iterrows():
        source_pair = (row["source_name"], row["index"])
        if source_pair in targets and targets[source_pair] is not None:
            all_channels.update(targets[source_pair].data_vars)

    # Add channels from masked source
    masked_pair_key = (masked_source, masked_idx)
    if masked_pair_key in targets and targets[masked_pair_key] is not None:
        all_channels.update(targets[masked_pair_key].data_vars)
    all_channels = sorted(list(all_channels))  # Sort for consistent ordering

    # Determine number of realizations to display
    num_realizations = 0
    masked_pair_key = (masked_source, masked_idx)
    if masked_pair_key in preds and preds[masked_pair_key] is not None:
        pred_ds = preds[masked_pair_key]
        if "realization" in pred_ds.dims:
            num_realizations = min(pred_ds.sizes["realization"], max_realizations_to_display)
        else:
            num_realizations = 1

    # Calculate number of rows (one per channel) and columns
    num_rows = len(all_channels)
    num_avail_sources = len(avail_pairs)
    # 1 column for channel labels + num_avail_sources + 1 separator + 1 target
    # + num_realizations predictions
    num_cols = 1 + num_avail_sources + 1 + 1 + num_realizations

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(3 * num_cols, 2.5 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.01, hspace=0.15)

    # Add channel labels on the left side
    for i, channel in enumerate(all_channels):
        ax = fig.add_subplot(gs[i, 0])
        ax.text(0.5, 0.5, channel, rotation=90, ha="center", va="center", fontweight="bold")
        ax.axis("off")

    # For each channel, plot available sources on the corresponding row
    for row_idx, channel in enumerate(all_channels):
        col_idx = 1
        for _, row in avail_pairs.iterrows():
            source_pair = (row["source_name"], row["index"])
            if source_pair in targets and targets[source_pair] is not None:
                target_ds = targets[source_pair]
                if channel in target_ds.data_vars:
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    target = target_ds[channel].values
                    # Crop NaN borders
                    target = crop_nan_border_numpy(target, [target])[0]
                    ax.imshow(target, cmap="viridis")

                    displayed_name = " ".join(row["source_name"].split("_")[3:5])
                    dt = row["dt"]
                    dt_str = strfdelta(dt, "%D d%H:%M")
                    title = f"{displayed_name}\n$\delta_t$={dt_str}"
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                else:
                    # Empty subplot if this source doesn't have this channel
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    ax.axis("off")
            col_idx += 1

        # Add separator column
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.axvline(x=0.5, color="black", linewidth=2)
        ax.axis("off")
        col_idx += 1

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

                    # Get all realizations for this channel
                    if "realization" in pred_ds.dims:
                        pred_realizations = pred_ds[
                            channel
                        ].values  # Shape: (realization, H, W) or (realization,)
                    else:
                        # Single realization case
                        pred_realizations = pred_ds[channel].values[
                            np.newaxis, ...
                        ]  # Add realization dimension

                    # Limit to max_realizations_to_display
                    pred_realizations = pred_realizations[:max_realizations_to_display]

                    # Crop NaN borders using only the target values for all data
                    all_data = [target] + [
                        pred_realizations[i] for i in range(len(pred_realizations))
                    ]
                    cropped_data = crop_nan_border_numpy(target, all_data)
                    target = cropped_data[0]
                    pred_realizations = cropped_data[1:]

                    # Calculate shared min/max values for this channel across target and all realizations
                    all_values = [target] + pred_realizations
                    vmin = min(np.nanmin(data) for data in all_values)
                    vmax = max(np.nanmax(data) for data in all_values)

                    # Plot masked source target
                    dt = sample_df[sample_df["source_name"] == masked_source]["dt"].iloc[0]
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    ax.imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)

                    displayed_name = " ".join(masked_source.split("_")[3:5])
                    dt_str = strfdelta(dt, "%D d%H:%M")
                    title = f"{displayed_name}\n$\delta_t$={dt_str} (Target)"
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                    col_idx += 1

                    # Plot all realizations (including predicted mean if enabled)
                    for real_idx in range(len(pred_realizations)):
                        # Check if this is the first realization and we want to show it as predicted mean
                        if real_idx == 0 and include_pred_mean:
                            title = f"{displayed_name}\nPredicted mean"
                        else:
                            if include_pred_mean:
                                realization_number = real_idx
                            else:
                                realization_number = real_idx + 1

                            title = f"{displayed_name}\nRealization {realization_number}"

                        ax = fig.add_subplot(gs[row_idx, col_idx])
                        ax.imshow(
                            pred_realizations[real_idx], cmap="viridis", vmin=vmin, vmax=vmax
                        )
                        ax.set_title(title, fontsize=8)
                        ax.axis("off")
                        col_idx += 1
                else:
                    # Empty subplots with message if channel not in predictions
                    for i in range(num_realizations + 1):  # +1 for target
                        ax = fig.add_subplot(gs[row_idx, col_idx])
                        ax.axis("off")
                        if i == 0:
                            displayed_name = " ".join(masked_source.split("_")[3:5])
                            ax.set_title(f"{displayed_name}\nTarget N/A", color="gray")
                        else:
                            ax.set_title(f"Realization {i}\nN/A", color="gray")
                        col_idx += 1
            else:
                # Empty subplots for target and predictions if masked source doesn't have this channel
                for i in range(num_realizations + 1):  # +1 for target
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    ax.axis("off")
                    col_idx += 1

    # Save the figure with tight layout and minimal padding
    plt.tight_layout(
        rect=[0.05, 0, 1, 0.95]
    )  # Adjust layout, leaving minimal space for row labels
    fig.savefig(
        results_dir / f"{batch_idx}_{sample_idx}.png", bbox_inches="tight", pad_inches=0.1, dpi=150
    )
    plt.close(fig)


def process_batch_chunk_multiple(
    chunk_df,
    batch_indices,
    load_batch_fn,
    results_dir,
    verbose=True,
    process_id=None,
    model_id=None,
    max_realizations_to_display=5,
    include_pred_mean=True,
):
    """Process a chunk of batches in parallel for multiple realizations visualization.

    Args:
        chunk_df (pd.DataFrame): DataFrame with batch information for this chunk
        batch_indices (list): List of batch indices to process
        load_batch_fn (callable): Function to load batch data
        results_dir (Path): Directory to save results
        verbose (bool): Whether to show progress
        process_id (int): ID of the process (for debugging)
        model_id (str): ID of the model
        max_realizations_to_display (int): Maximum number of realizations to display
        include_pred_mean (bool): If True, expects the first realization in preds to be
            the predicted mean. It is then displayed as the first column after the target.
    """
    for batch_idx in tqdm(
        batch_indices, desc=f"Worker {process_id}", disable=not verbose, position=process_id
    ):
        batch_info = chunk_df[chunk_df["batch_idx"] == batch_idx]

        # For each source, load the targets and predictions
        targets, preds = {}, {}
        for _, row in batch_info.iterrows():
            src, index = row["source_name"], row["index"]
            targets[(src, index)], preds[(src, index)] = load_batch_fn(
                src, index, batch_idx, model_id
            )

        # Display the targets and predictions
        display_batch_multiple(
            batch_info,
            batch_idx,
            targets,
            preds,
            results_dir,
            model_id,
            max_realizations_to_display,
            include_pred_mean=include_pred_mean,
        )

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
