"""
Implements the VisualEvaluationComparison class, which displays the targets and predictions
from multiple models for a given source in a single figure. This visualization is useful
for comparing the predictions of different models for the same sample.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from string import Template

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from multi_sources.data_processing.grid_functions import crop_nan_border_numpy
from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


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
    """Creates comparison visualizations with available sources in the top row
    and model predictions (realizations) in subsequent rows."""

    def __init__(
        self,
        model_data,
        parent_results_dir,
        eval_fraction=1.0,
        max_realizations_to_display=6,
    ):
        super().__init__(
            "visual_eval_comparison",
            "Comparison visualization of multi-model predictions",
            model_data,
            parent_results_dir,
        )
        self.eval_fraction = eval_fraction
        self.max_realizations_to_display = max_realizations_to_display

    def evaluate_sources(self, verbose=True, num_workers=0):
        """Main evaluation method."""
        # Get batch indices from first model (assuming all models have the same indices)
        first_model_data = next(iter(self.model_data.values()))
        batch_indices = sorted(first_model_data["info_df"]["batch_idx"].unique())

        # Subset if eval_fraction < 1.0
        if self.eval_fraction < 1.0:
            num_eval = int(np.ceil(self.eval_fraction * len(batch_indices)))
            batch_indices = np.random.choice(batch_indices, num_eval, replace=False)

        # Create results directory
        results_dir = self.parent_results_dir / "model_comparison"
        results_dir.mkdir(exist_ok=True, parents=True)

        print(f"Creating visualizations for {len(self.model_data)} models")

        # Process batches
        if num_workers < 2:
            self._process_batches_sequential(batch_indices, results_dir, verbose)
        else:
            self._process_batches_parallel(batch_indices, results_dir, num_workers, verbose)

    def _process_batches_sequential(self, batch_indices, results_dir, verbose):
        """Process batches sequentially."""
        for batch_idx in tqdm(batch_indices, desc="Batches", disable=not verbose):
            batch_data = self._collect_batch_data(batch_idx)
            self._create_batch_visualizations(batch_idx, batch_data, results_dir)

    def _process_batches_parallel(self, batch_indices, results_dir, num_workers, verbose):
        """Process batches in parallel."""
        chunk_size = max(1, len(batch_indices) // num_workers)
        chunks = [
            batch_indices[i : i + chunk_size] for i in range(0, len(batch_indices), chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_batch_chunk,
                    chunk,
                    self.model_data,
                    self.load_batch,
                    results_dir,
                    verbose and i == 0,
                    i,
                    self.max_realizations_to_display,
                )
                for i, chunk in enumerate(chunks)
            ]
            for future in futures:
                future.result()

    def _collect_batch_data(self, batch_idx):
        """Collect data for all models for a given batch."""
        batch_data = {}
        for model_id, data in self.model_data.items():
            info_df = data["info_df"]
            batch_info = info_df[info_df["batch_idx"] == batch_idx]

            targets, preds = {}, {}
            for _, row in batch_info.iterrows():
                src, index = row["source_name"], row["source_index"]
                targets[(src, index)], preds[(src, index)] = self.load_batch(
                    src, index, batch_idx, model_id
                )

            batch_data[model_id] = {
                "info_df": batch_info,
                "targets": targets,
                "preds": preds,
            }
        return batch_data

    def _create_batch_visualizations(self, batch_idx, batch_data, results_dir):
        """Create visualizations for all samples in a batch."""
        first_model_id = list(batch_data.keys())[0]
        sample_indices = batch_data[first_model_id]["info_df"]["index_in_batch"].unique()

        for sample_idx in sample_indices:
            sample_data = self._extract_sample_data(batch_data, sample_idx)
            create_sample_visualization(
                sample_data, results_dir, batch_idx, sample_idx, self.max_realizations_to_display
            )

    def _extract_sample_data(self, batch_data, sample_idx):
        """Extract data for a specific sample from batch data."""
        sample_data = {}
        for model_id, data in batch_data.items():
            sample_df = data["info_df"][data["info_df"]["index_in_batch"] == sample_idx]

            sample_targets, sample_preds = {}, {}
            for _, row in sample_df.iterrows():
                src, index = row["source_name"], row["source_index"]
                source_pair = (src, index)

                if source_pair in data["targets"] and data["targets"][source_pair] is not None:
                    sample_targets[source_pair] = data["targets"][source_pair].isel(
                        samples=sample_idx
                    )
                if source_pair in data["preds"] and data["preds"][source_pair] is not None:
                    sample_preds[source_pair] = data["preds"][source_pair].isel(samples=sample_idx)

            sample_data[model_id] = {
                "info_df": sample_df,
                "targets": sample_targets,
                "preds": sample_preds,
            }
        return sample_data


def process_batch_chunk(
    batch_indices,
    model_data,
    load_batch_fn,
    results_dir,
    show_progress,
    worker_idx,
    max_realizations_to_display,
):
    """Process a chunk of batch indices in parallel."""
    for batch_idx in tqdm(batch_indices, desc=f"Worker {worker_idx}", disable=not show_progress):
        # Collect batch data
        batch_data = {}
        for model_id, data in model_data.items():
            info_df = data["info_df"]
            batch_info = info_df[info_df["batch_idx"] == batch_idx]

            targets, preds = {}, {}
            for _, row in batch_info.iterrows():
                src, index = row["source_name"], row["source_index"]
                targets[(src, index)], preds[(src, index)] = load_batch_fn(
                    src, index, batch_idx, model_id
                )

            batch_data[model_id] = {
                "info_df": batch_info,
                "targets": targets,
                "preds": preds,
            }

        # Create visualizations
        first_model_id = list(batch_data.keys())[0]
        sample_indices = batch_data[first_model_id]["info_df"]["index_in_batch"].unique()

        for sample_idx in sample_indices:
            sample_data = extract_sample_data(batch_data, sample_idx)
            create_sample_visualization(
                sample_data, results_dir, batch_idx, sample_idx, max_realizations_to_display
            )


def extract_sample_data(batch_data, sample_idx):
    """Extract data for a specific sample from batch data."""
    sample_data = {}
    for model_id, data in batch_data.items():
        sample_df = data["info_df"][data["info_df"]["index_in_batch"] == sample_idx]

        sample_targets, sample_preds = {}, {}
        for _, row in sample_df.iterrows():
            src, index = row["source_name"], row["source_index"]
            source_pair = (src, index)

            if source_pair in data["targets"] and data["targets"][source_pair] is not None:
                sample_targets[source_pair] = data["targets"][source_pair].isel(samples=sample_idx)
            if source_pair in data["preds"] and data["preds"][source_pair] is not None:
                sample_preds[source_pair] = data["preds"][source_pair].isel(samples=sample_idx)

        sample_data[model_id] = {
            "info_df": sample_df,
            "targets": sample_targets,
            "preds": sample_preds,
        }
    return sample_data


def get_first_channel(sample_data):
    """Get the first available channel from the data."""
    for model_data in sample_data.values():
        for target_ds in model_data["targets"].values():
            if target_ds is not None and len(target_ds.data_vars) > 0:
                return list(target_ds.data_vars)[0]
    return None


def get_available_sources(sample_data):
    """Get available sources and targets sorted by decreasing dt."""
    first_model_id = list(sample_data.keys())[0]
    first_model_info = sample_data[first_model_id]["info_df"]
    # Include both available sources (avail == 1) and target sources (avail == 0)
    sources_and_targets = first_model_info[first_model_info["avail"].isin([0, 1])]
    return sources_and_targets.sort_values(by="dt", ascending=False)


def get_global_color_limits(sample_data, first_channel):
    """Calculate global min/max for consistent color scaling."""
    vmin, vmax = float("inf"), float("-inf")
    pred_mean_channel = f"pred_mean_{first_channel}"

    def update_limits(data):
        nonlocal vmin, vmax
        if data is not None and not np.all(np.isnan(data)):
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                vmin = min(vmin, np.nanmin(valid_data))
                vmax = max(vmax, np.nanmax(valid_data))

    for model_data in sample_data.values():
        # Check targets
        for target_ds in model_data["targets"].values():
            if target_ds is not None and first_channel in target_ds.data_vars:
                update_limits(target_ds[first_channel].values)

        # Check predictions
        for pred_ds in model_data["preds"].values():
            if pred_ds is not None:
                # Check regular channel predictions
                if first_channel in pred_ds.data_vars:
                    if "realization" in pred_ds.dims:
                        for r in range(pred_ds.sizes["realization"]):
                            update_limits(pred_ds[first_channel].isel(realization=r).values)
                    else:
                        update_limits(pred_ds[first_channel].values)

                # Check predicted mean channel
                if pred_mean_channel in pred_ds.data_vars:
                    update_limits(pred_ds[pred_mean_channel].values)

    return (0, 1) if vmin == float("inf") or vmax == float("-inf") else (vmin, vmax)


def plot_available_sources(
    fig,
    avail_pairs,
    sample_data,
    first_channel,
    vmin,
    vmax,
    max_avail_cols,
    num_models,
    total_cols,
):
    """Plot available sources and targets in the top row."""
    first_model_id = list(sample_data.keys())[0]

    for i, (_, row) in enumerate(avail_pairs.iterrows()):
        if i >= max_avail_cols:  # Limit columns
            break

        src, idx = row["source_name"], row["source_index"]
        source_pair = (src, idx)
        dt_str = strfdelta(row["dt"], "%D days %H:%M")

        # Add label to distinguish between available sources and targets
        label = "Target" if row["avail"] == 0 else "Available"

        if source_pair in sample_data[first_model_id]["targets"]:
            target_ds = sample_data[first_model_id]["targets"][source_pair]
            if target_ds is not None and first_channel in target_ds.data_vars:
                data = target_ds[first_channel].values
                data = crop_nan_border_numpy(data, [data])[0]

                ax = fig.add_subplot(1 + num_models, total_cols, i + 1)  # Top row
                ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(f"{src} ({dt_str})\n{label}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])


def plot_model_predictions(
    fig, sample_data, first_channel, vmin, vmax, max_realizations, max_cols, num_models
):
    """Plot model predictions - one row per model - only for target sources (avail == 0)."""
    model_ids = list(sample_data.keys())

    # Find the target source (avail == 0) - this is what we want to predict
    first_model_info = sample_data[model_ids[0]]["info_df"]
    target_pairs = first_model_info[first_model_info["avail"] == 0]
    if len(target_pairs) == 0:
        return None  # No target source found

    target_pair = target_pairs.iloc[0]  # Use first target source
    src, idx = target_pair["source_name"], target_pair["source_index"]
    source_pair = (src, idx)
    pred_mean_channel = f"pred_mean_{first_channel}"

    im = None
    # Create one row per model (starting from row 2, since row 1 is for available sources)
    for model_idx, model_id in enumerate(model_ids):
        model_data = sample_data[model_id]
        row_idx = model_idx + 2  # Start from row 2 (1-indexed for subplot)

        # Check if predicted mean is available for this model
        has_pred_mean = (
            source_pair in model_data["preds"]
            and model_data["preds"][source_pair] is not None
            and pred_mean_channel in model_data["preds"][source_pair].data_vars
        )

        # Plot predictions for this model across columns
        for col_idx in range(max_cols):
            ax = fig.add_subplot(1 + num_models, max_cols, (row_idx - 1) * max_cols + col_idx + 1)

            if source_pair in model_data["preds"] and model_data["preds"][source_pair] is not None:
                pred_ds = model_data["preds"][source_pair]

                # First column: plot predicted mean if available, otherwise first realization
                if col_idx == 0 and has_pred_mean:
                    # Plot predicted mean
                    data = pred_ds[pred_mean_channel].values

                    # Get target for cropping reference
                    if source_pair in model_data["targets"]:
                        target_data = model_data["targets"][source_pair][first_channel].values
                        data = crop_nan_border_numpy(target_data, [data])[0]

                    im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title(f"Model: {model_id}\nPred. Mean", fontsize=9)

                elif first_channel in pred_ds.data_vars:
                    # Calculate realization index, accounting for predicted mean in first column
                    real_idx = col_idx - (1 if has_pred_mean else 0)

                    # Skip if we've exceeded available realizations
                    if real_idx >= max_realizations:
                        ax.axis("off")
                        continue

                    # Get data for this realization
                    if "realization" in pred_ds.dims:
                        if real_idx >= 0 and real_idx < pred_ds.sizes["realization"]:
                            data = pred_ds[first_channel].isel(realization=real_idx).values
                        else:
                            # Empty subplot if no realization available
                            ax.axis("off")
                            continue
                    else:
                        if real_idx == 0:  # Only show single prediction in first realization slot
                            data = pred_ds[first_channel].values
                        else:
                            # Empty subplot for other columns
                            ax.axis("off")
                            continue

                    # Get target for cropping reference
                    if source_pair in model_data["targets"]:
                        target_data = model_data["targets"][source_pair][first_channel].values
                        data = crop_nan_border_numpy(target_data, [data])[0]

                    im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)

                    # Add title
                    if col_idx == 0 and not has_pred_mean:
                        ax.set_title(f"Model: {model_id}\nReal. {real_idx + 1}", fontsize=9)
                    else:
                        ax.set_title(f"Real. {real_idx + 1}", fontsize=9)
                else:
                    # Empty subplot if no predictions for this channel
                    ax.axis("off")
            else:
                # Empty subplot if no predictions for this source
                ax.axis("off")
                if col_idx == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"Model: {model_id}\n(No predictions)",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                    )

            ax.set_xticks([])
            ax.set_yticks([])

    return im  # Return for colorbar


def create_sample_visualization(sample_data, results_dir, batch_idx, sample_idx, max_realizations):
    """Create and save a visualization for a single sample."""
    try:
        # Get basic info
        first_channel = get_first_channel(sample_data)
        if first_channel is None:
            print(f"Warning: No valid channel found for batch {batch_idx}, sample {sample_idx}")
            return

        avail_pairs = get_available_sources(sample_data)
        vmin, vmax = get_global_color_limits(sample_data, first_channel)

        # Calculate grid dimensions
        num_models = len(sample_data)

        # For available sources row: limit to available sources or max_realizations
        max_avail_cols = min(len(avail_pairs), max_realizations)

        # For prediction rows: check if any model has predicted means to determine max columns needed
        max_pred_cols = max_realizations
        for model_data in sample_data.values():
            for pred_ds in model_data["preds"].values():
                if pred_ds is not None:
                    pred_mean_channel = f"pred_mean_{first_channel}"
                    if pred_mean_channel in pred_ds.data_vars:
                        # Add 1 for predicted mean column
                        max_pred_cols = max_realizations + 1
                        break
            if max_pred_cols > max_realizations:
                break

        # Use the maximum of both requirements
        max_cols = max(max_avail_cols, max_pred_cols)

        # Total rows: 1 for available sources + 1 per model
        total_rows = 1 + num_models

        # Create figure - adjust width to account for colorbar
        fig_width = 3 * max_cols + 1.5  # Add more space for colorbar
        fig_height = 2.5 * total_rows  # Adjust height per row
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Plot available sources (top row)
        plot_available_sources(
            fig,
            avail_pairs,
            sample_data,
            first_channel,
            vmin,
            vmax,
            max_avail_cols,
            num_models,
            max_cols,
        )

        # Plot model predictions (one row per model)
        im = plot_model_predictions(
            fig, sample_data, first_channel, vmin, vmax, max_realizations, max_cols, num_models
        )

        # Add vertical colorbar on the right if we have an image
        if im is not None:
            # Use plt.subplots_adjust to make room for colorbar
            plt.subplots_adjust(right=0.85)
            fig.colorbar(
                im,
                ax=fig.get_axes(),
                orientation="vertical",
                fraction=0.05,
                pad=0.02,
                shrink=0.8,
            )

        else:
            plt.tight_layout()

        # Save figure
        if not isinstance(results_dir, Path):
            results_dir = Path(results_dir)

        img_path = results_dir / f"batch_{batch_idx}_sample_{sample_idx}_comparison.png"
        plt.savefig(img_path, dpi=150, bbox_inches="tight")

        pdf_path = results_dir / f"batch_{batch_idx}_sample_{sample_idx}_comparison.pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    finally:
        plt.close("all")
