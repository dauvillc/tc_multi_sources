"""
Implements the VisualEvaluationComparison class, which displays the targets and predictions
from multiple models for a given source in a single figure. This visualization is useful
for comparing the predictions of different models for the same sample.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path  # Used for path operations with / operator
from string import Template

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
    d["M"], _ = divmod(rem, 60)
    d["H"], d["M"] = f"{d['H']:02d}", f"{d['M']:02d}"
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


class VisualEvaluationComparison(AbstractMultisourceEvaluationMetric):
    """Displays the targets and all realizations of predictions from multiple models for a given source.
    For each sample in each batch:
    - Retrieves the list of sources that were included in the batch.
    - Loads the targets and predictions for each model.
    - Creates a figure with two parts:
        - Top part: Available sources (first channel) on one line, masked sources on another line
        - Bottom part: One line per model, with each model's realizations as rows
    - Saves the figure to the results directory.
    """

    def __init__(
        self,
        model_data,
        parent_results_dir,
        eval_fraction=1.0,
        max_realizations_to_display=6,
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
        """
        super().__init__(
            "visual_eval_comparison",
            "Comparison visualization of multi-model predictions",
            model_data,
            parent_results_dir,
        )
        self.eval_fraction = eval_fraction
        self.max_realizations_to_display = max_realizations_to_display

    def evaluate_sources(self, verbose=True, num_workers=0):
        """
        Args:
            verbose (bool): Whether to show progress bars.
            num_workers (int): Number of workers for parallel processing.
        """
        sns.set_context("paper")

        # We need to find batch indices that are common across all models
        common_batch_indices = None

        for model_id, data in self.model_data.items():
            info_df = data["info_df"]
            batch_indices = info_df["batch_idx"].unique()

            if common_batch_indices is None:
                common_batch_indices = set(batch_indices)
            else:
                common_batch_indices = common_batch_indices.intersection(batch_indices)

        common_batch_indices = sorted(list(common_batch_indices))

        if not common_batch_indices:
            print(
                "No common batch indices found across models. Cannot create comparison visualizations."
            )
            return

        # If eval_fraction < 1.0, select a random subset of the batch indices
        if self.eval_fraction < 1.0:
            num_batches = len(common_batch_indices)
            num_eval_batches = np.ceil(self.eval_fraction * num_batches).astype(int)
            common_batch_indices = np.random.choice(
                common_batch_indices, num_eval_batches, replace=False
            )

        print(
            f"\nCreating multi-model comparison visualizations for {len(self.model_data)} models"
        )

        # Create results directory specifically for comparison visualizations
        comparison_results_dir = self.parent_results_dir / "model_comparison"
        comparison_results_dir.mkdir(exist_ok=True, parents=True)

        if num_workers < 2:
            for batch_idx in tqdm(common_batch_indices, desc="Batches", disable=not verbose):
                # Collect data from all models for this batch
                model_batch_data = {}

                for model_id, data in self.model_data.items():
                    info_df = data["info_df"]
                    batch_info = info_df[info_df["batch_idx"] == batch_idx]

                    # For each source, load the targets and predictions
                    targets, preds = {}, {}
                    for _, row in batch_info.iterrows():
                        src, index = row["source_name"], row["index"]
                        targets[(src, index)], preds[(src, index)] = self.load_batch(
                            src, index, batch_idx, model_id
                        )

                    model_batch_data[model_id] = {
                        "info_df": batch_info,
                        "targets": targets,
                        "preds": preds,
                    }

                # Display the comparison visualization for this batch
                display_batch_comparison(
                    batch_idx,
                    model_batch_data,
                    comparison_results_dir,
                    self.max_realizations_to_display,
                )
        else:
            # Use parallel processing
            chunk_size = max(1, len(common_batch_indices) // num_workers)
            chunks = [
                common_batch_indices[i : i + chunk_size]
                for i in range(0, len(common_batch_indices), chunk_size)
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    future = executor.submit(
                        process_batch_chunk_comparison,
                        chunk,
                        self.model_data,
                        self.load_batch,
                        comparison_results_dir,
                        verbose and i == 0,  # Only show progress for the first worker
                        i,
                        self.max_realizations_to_display,
                    )
                    futures.append(future)

                # Wait for all workers to complete
                for future in futures:
                    future.result()


def process_batch_chunk_comparison(
    batch_indices,
    model_data,
    load_batch_fn,
    results_dir,
    show_progress,
    worker_idx,
    max_realizations_to_display,
):
    """Process a chunk of batch indices in parallel.

    Args:
        batch_indices (list): List of batch indices to process
        model_data (dict): Dictionary mapping model_ids to dictionaries of model data
        load_batch_fn (function): Function to load batch data
        results_dir (Path): Directory to save results
        show_progress (bool): Whether to show progress bar
        worker_idx (int): Worker index for identification
        max_realizations_to_display (int): Maximum number of realizations to display
    """
    for batch_idx in tqdm(batch_indices, desc=f"Worker {worker_idx}", disable=not show_progress):
        # Collect data from all models for this batch
        model_batch_data = {}

        for model_id, data in model_data.items():
            info_df = data["info_df"]
            batch_info = info_df[info_df["batch_idx"] == batch_idx]

            # For each source, load the targets and predictions
            targets, preds = {}, {}
            for _, row in batch_info.iterrows():
                src, index = row["source_name"], row["index"]
                targets[(src, index)], preds[(src, index)] = load_batch_fn(
                    src, index, batch_idx, model_id
                )

            model_batch_data[model_id] = {
                "info_df": batch_info,
                "targets": targets,
                "preds": preds,
            }

        # Display the comparison visualization for this batch
        display_batch_comparison(
            batch_idx,
            model_batch_data,
            results_dir,
            max_realizations_to_display,
        )


def display_batch_comparison(
    batch_idx,
    model_batch_data,
    results_dir,
    max_realizations_to_display,
):
    """Displays the targets and predictions from multiple models for a given batch index.

    Args:
        batch_idx (int): Index of the batch to display
        model_batch_data (dict): Dictionary mapping model_ids to batch data
        results_dir (Path): Directory where results will be saved
        max_realizations_to_display (int): Maximum number of realizations to display
    """
    # Get a representative model to determine batch structure
    first_model_id = list(model_batch_data.keys())[0]
    first_model_info = model_batch_data[first_model_id]["info_df"]

    batch_indices = first_model_info["index_in_batch"].unique()

    try:
        for idx in batch_indices:
            # Collect sample data from all models
            model_sample_data = {}

            for model_id, data in model_batch_data.items():
                batch_info = data["info_df"]
                sample_df = batch_info[batch_info["index_in_batch"] == idx]

                # For each source, select only the targets and predictions for the sample
                sample_targets, sample_preds = {}, {}
                for _, row in sample_df.iterrows():
                    src, index = row["source_name"], row["index"]
                    source_pair = (src, index)
                    if source_pair in data["targets"] and source_pair in data["preds"]:
                        if data["targets"][source_pair] is not None:
                            sample_targets[source_pair] = data["targets"][source_pair].isel(
                                samples=idx
                            )
                        if data["preds"][source_pair] is not None:
                            sample_preds[source_pair] = data["preds"][source_pair].isel(
                                samples=idx
                            )

                model_sample_data[model_id] = {
                    "info_df": sample_df,
                    "targets": sample_targets,
                    "preds": sample_preds,
                }

            plot_sample_comparison(
                model_sample_data,
                results_dir,
                batch_idx,
                idx,
                max_realizations_to_display,
            )
    finally:
        # Make sure to close all figures even if an error occurs
        plt.close("all")


def apply_post_processing(target_ds, pred_ds, source_name=None):
    """Apply post-processing to target and prediction datasets.

    Args:
        target_ds (xarray.Dataset): Target dataset
        pred_ds (xarray.Dataset): Prediction dataset
        source_name (str): Name of the source

    Returns:
        tuple: Processed target and prediction datasets
    """
    # If no predictions, return the targets as is
    if pred_ds is None:
        return target_ds, pred_ds

    # Apply post-processing here if needed

    return target_ds, pred_ds


def plot_sample_comparison(
    model_sample_data,
    results_dir,
    batch_idx,
    sample_idx,
    max_realizations_to_display,
):
    """Creates a comparison visualization with Matplotlib, showing:
    - Top part: Available sources (first channel) on one line, masked sources on another line
    - Bottom part: One line per model, with each model's realizations as rows

    Args:
        model_sample_data (dict): Dictionary mapping model_ids to sample data
        results_dir (Path): Directory where results will be saved
        batch_idx (int): Batch index
        sample_idx (int): Sample index within the batch
        max_realizations_to_display (int): Maximum number of realizations to display
    """
    # Ensure results_dir is a Path object
    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    # Get a representative model to determine sample structure
    first_model_id = list(model_sample_data.keys())[0]
    first_model_info = model_sample_data[first_model_id]["info_df"]

    # Retrieve available and masked sources
    avail_pairs = first_model_info[first_model_info["avail"] == 1]
    masked_pairs = first_model_info[first_model_info["avail"] == 0]

    if len(masked_pairs) == 0:
        return  # Skip if no masked source is found

    # Process all sources for all models
    for model_id, data in model_sample_data.items():
        for _, row in data["info_df"].iterrows():
            source_name, index = row["source_name"], row["index"]
            source_pair = (source_name, index)
            if source_pair in data["targets"] and data["targets"][source_pair] is not None:
                # Apply post-processing
                data["targets"][source_pair], data["preds"][source_pair] = apply_post_processing(
                    data["targets"][source_pair], data["preds"].get(source_pair), source_name
                )

    # Sort available pairs by decreasing dt
    avail_pairs = avail_pairs.sort_values(by="dt", ascending=False)

    # Get the first channel from the first available source to use as reference
    first_channel = None
    for _, row in avail_pairs.iterrows():
        src, idx = row["source_name"], row["index"]
        source_pair = (src, idx)
        if source_pair in model_sample_data[first_model_id]["targets"]:
            target_ds = model_sample_data[first_model_id]["targets"][source_pair]
            if target_ds is not None and len(target_ds.data_vars) > 0:
                first_channel = list(target_ds.data_vars)[0]
                break

    if first_channel is None:
        print(f"Warning: No valid channel found for batch {batch_idx}, sample {sample_idx}")
        return

    # Count max number of realizations across models
    max_reals = 0
    for model_id, data in model_sample_data.items():
        for masked_pair in masked_pairs.iterrows():
            src, idx = masked_pair[1]["source_name"], masked_pair[1]["index"]
            source_pair = (src, idx)
            if source_pair in data["preds"] and data["preds"][source_pair] is not None:
                pred_ds = data["preds"][source_pair]
                if "realization" in pred_ds.dims:
                    num_reals = min(pred_ds.sizes["realization"], max_realizations_to_display)
                    max_reals = max(max_reals, num_reals)
                else:
                    max_reals = max(max_reals, 1)

    # Calculate grid dimensions
    num_avail_sources = len(avail_pairs)
    num_masked_sources = len(masked_pairs)
    num_models = len(model_sample_data)

    # Create figure with an irregular grid
    max_cols = max(num_avail_sources, num_masked_sources)
    # Rows: 2 for available/masked sources + rows for each model's realizations
    num_rows = 2 + sum(max_reals for _ in range(num_models))

    # Set figure size (adjust as needed)
    fig_width = 2.5 * max_cols  # Width per subplot in inches
    fig_height = 1.5 * num_rows  # Height per subplot in inches

    # Create figure and gridspec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        num_rows, max_cols, hspace=0.1, wspace=0.05, left=0.1
    )  # Reduced spacing between images

    # Find global min/max values for consistent colormap scaling
    vmin, vmax = float("inf"), float("-inf")

    # Function to update vmin/vmax from an array
    def update_limits(data):
        nonlocal vmin, vmax
        if data is not None and not np.all(np.isnan(data)):
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                vmin = min(vmin, np.nanmin(valid_data))
                vmax = max(vmax, np.nanmax(valid_data))

    # Scan all data to find global limits
    for model_id, data in model_sample_data.items():
        # Check all available and masked source pairs
        for _, row in data["info_df"].iterrows():
            source_pair = (row["source_name"], row["index"])
            if source_pair in data["targets"] and data["targets"][source_pair] is not None:
                if first_channel in data["targets"][source_pair].data_vars:
                    update_limits(data["targets"][source_pair][first_channel].values)

            if source_pair in data["preds"] and data["preds"][source_pair] is not None:
                if first_channel in data["preds"][source_pair].data_vars:
                    if "realization" in data["preds"][source_pair].dims:
                        for r in range(data["preds"][source_pair].sizes["realization"]):
                            update_limits(
                                data["preds"][source_pair][first_channel]
                                .isel(realization=r)
                                .values
                            )
                    else:
                        update_limits(data["preds"][source_pair][first_channel].values)

    # If we didn't find valid min/max, set defaults
    if vmin == float("inf") or vmax == float("-inf"):
        vmin, vmax = 0, 1

    # Top part - Available sources (first row)
    for i, (_, row) in enumerate(avail_pairs.iterrows()):
        if i >= max_cols:  # Skip if we have more sources than columns
            continue

        src, idx = row["source_name"], row["index"]
        source_pair = (src, idx)
        dt_str = strfdelta(row["dt"], "%D days %H:%M")

        if source_pair in model_sample_data[first_model_id]["targets"]:
            target_ds = model_sample_data[first_model_id]["targets"][source_pair]
            if target_ds is not None and first_channel in target_ds.data_vars:
                target_data = target_ds[first_channel].values
                if target_data is not None:
                    # Remove NaN borders if present
                    target_data = crop_nan_border_numpy(target_data, [target_data])[0]

                    # Create subplot at position (0, i)
                    ax = fig.add_subplot(gs[0, i])
                    im = ax.imshow(target_data, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title(f"{src} ({dt_str})", fontsize=8, pad=2)  # Reduced padding
                    ax.set_xticks([])
                    ax.set_yticks([])

    # Masked sources (second row)
    for i, (_, row) in enumerate(masked_pairs.iterrows()):
        if i >= max_cols:  # Skip if we have more sources than columns
            continue

        src, idx = row["source_name"], row["index"]
        source_pair = (src, idx)
        dt_str = strfdelta(row["dt"], "%D days %H:%M")

        if source_pair in model_sample_data[first_model_id]["targets"]:
            target_ds = model_sample_data[first_model_id]["targets"][source_pair]
            if target_ds is not None and first_channel in target_ds.data_vars:
                target_data = target_ds[first_channel].values
                if target_data is not None:
                    # Remove NaN borders if present
                    target_data = crop_nan_border_numpy(target_data, [target_data])[0]

                    # Create subplot at position (1, i)
                    ax = fig.add_subplot(gs[1, i])
                    im = ax.imshow(target_data, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title(
                        f"{src} ({dt_str}) - Target", fontsize=8, pad=2
                    )  # Reduced padding
                    ax.set_xticks([])
                    ax.set_yticks([])

    # Bottom part - Model predictions
    current_row = 2  # Start from the third row (after available and masked sources)

    for model_idx, (model_id, data) in enumerate(model_sample_data.items()):

        # For each masked source
        for masked_idx, (_, row) in enumerate(masked_pairs.iterrows()):
            if masked_idx >= max_cols:
                continue

            src, idx = row["source_name"], row["index"]
            source_pair = (src, idx)

            if source_pair in data["preds"] and data["preds"][source_pair] is not None:
                pred_ds = data["preds"][source_pair]

                # Check if predictions have realizations dimension
                if "realization" in pred_ds.dims:
                    num_reals = min(pred_ds.sizes["realization"], max_realizations_to_display)

                    # Plot each realization in its own row
                    for real_idx in range(num_reals):
                        if current_row + real_idx >= num_rows:
                            break

                        if first_channel in pred_ds.data_vars:
                            # Extract prediction for this realization
                            pred_data = pred_ds[first_channel].isel(realization=real_idx).values

                            # Remove NaN borders if present
                            pred_data = crop_nan_border_numpy(pred_data, [pred_data])[0]

                            # Create subplot
                            ax = fig.add_subplot(gs[current_row + real_idx, masked_idx])
                            im = ax.imshow(pred_data, cmap="viridis", vmin=vmin, vmax=vmax)

                            # Add model name as title for the first subplot in each model group
                            if masked_idx == 0 and real_idx == 0:
                                ax.set_title(
                                    f"Model: {model_id}", fontsize=9, fontweight="bold", pad=2
                                )

                            # Add small realization number indicator for all realizations except the first one in first column
                            if real_idx > 0:
                                ax.text(
                                    2,
                                    5,
                                    f"R{real_idx+1}",
                                    fontsize=7,
                                    color="white",
                                    backgroundcolor="black",
                                    bbox=dict(
                                        facecolor="black", alpha=0.7, boxstyle="round,pad=0.1"
                                    ),
                                )

                            ax.set_xticks([])
                            ax.set_yticks([])
                else:
                    # Single prediction without realizations
                    if first_channel in pred_ds.data_vars:
                        pred_data = pred_ds[first_channel].values
                        pred_data = crop_nan_border_numpy(pred_data, [pred_data])[0]

                        # Create subplot
                        ax = fig.add_subplot(gs[current_row, masked_idx])
                        im = ax.imshow(pred_data, cmap="viridis", vmin=vmin, vmax=vmax)

                        # Add model name as title for the first subplot
                        if masked_idx == 0:
                            ax.set_title(f"Model: {model_id}", fontsize=9, fontweight="bold")

                        ax.set_xticks([])
                        ax.set_yticks([])

        # Move to the next set of rows for the next model
        current_row += max_reals

    # Add a colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

    # Adjust layout
    plt.tight_layout(rect=[0.1, 0.05, 1, 0.98])  # More compact layout with less padding

    # Save the figure
    img_path = results_dir / f"batch_{batch_idx}_sample_{sample_idx}_comparison.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")

    # Also save as PDF for vector graphics
    pdf_path = results_dir / f"batch_{batch_idx}_sample_{sample_idx}_comparison.pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
