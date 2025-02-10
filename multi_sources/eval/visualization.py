"""
Implements the VisualEvaluation class, which just displays the targets and predictions
for a given source.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from string import Template
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric
import matplotlib.gridspec as gridspec


# A little code snippet from Shawn Chin & Peter Mortensen
# https://stackoverflow.com/questions/8906926/formatting-timedelta-objects/8907269#8907269
class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    """equivalent of strftime for timedelta objects"""
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    d["H"], d["M"], d["S"] = f'{d["H"]:02d}', f'{d["M"]:02d}', f'{d["S"]:02d}'
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
    # For each sample, create a figure with the targets and predictions
    batch_indices = batch_info["index_in_batch"].unique()
    for idx in batch_indices:
        sample_df = batch_info[batch_info["index_in_batch"] == idx]
        # For each source, select only the targets and predictions for the sample
        sample_targets = {source: targets[source].isel(samples=idx) for source in sources}
        sample_preds = {source: preds[source].isel(samples=idx) for source in sources}
        plot_sample(sample_df, sample_targets, sample_preds, results_dir, batch_idx, idx)


def plot_sample(sample_df, targets, preds, results_dir, batch_idx, sample_idx):
    """Displays in a figure with a single row. From left to right: available sources,
    masked source target, and prediction."""
    # Retrieve the list of all sources that are available and not masked
    avail_sources = sample_df[sample_df["avail"] == 1]["source_name"].unique().tolist()
    # Retrieve the source that is masked
    masked_source = sample_df[sample_df["avail"] == 0]["source_name"].values[0]

    # Sort sources by decreasing dt
    source_dt = {row["source_name"]: row["dt"] for _, row in sample_df.iterrows()}
    avail_sources.sort(key=lambda x: source_dt[x], reverse=True)

    # Calculate total number of plots needed
    num_avail = len(avail_sources)
    total_cols = num_avail + 2  # available sources + masked target + prediction

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(3 * total_cols + 1, 3.5))  # Slightly taller to accommodate labels
    gs = gridspec.GridSpec(1, total_cols + 1, wspace=0.4)
    # Adjust the subplot parameters to give specified padding
    plt.subplots_adjust(top=0.75, bottom=0.15)  # Increased top margin for labels

    # Add section labels - adjust the x-position for "Available sources" to center it over its columns
    fig.text(
        num_avail / (2 * total_cols),
        0.92,  # Changed from 0.9 to 0.92
        "Available sources",
        fontsize=14,
        weight="bold",
        ha="center",
    )
    fig.text(
        (total_cols - 0.15) / (total_cols + 1),
        0.92,  # Changed from 0.9 to 0.92
        "Target / Prediction",
        fontsize=14,
        weight="bold",
        ha="center",
    )

    # Plot available sources
    for i, source in enumerate(avail_sources):
        ax = fig.add_subplot(gs[0, i])
        target_ds = targets[source]
        first_channel = list(target_ds.data_vars.keys())[0]
        target = target_ds[first_channel].values
        dt = sample_df[sample_df["source_name"] == source]["dt"].iloc[0]
        ax.imshow(target, cmap="viridis")

        displayed_name = " ".join(source.split("_")[3:5])
        displayed_name += " - " + first_channel
        title = f"{displayed_name}\n$\delta_t=${strfdelta(dt, '%H:%M:%S')}"
        ax.set_title(title)
        set_axis_ticks(ax, target_ds.lat.values, target_ds.lon.values)

    # Add vertical separator line
    fig.add_subplot(gs[0, num_avail])
    plt.axvline(x=0, color="black", linestyle="-", linewidth=2)
    plt.axis("off")

    # Get target and prediction data for masked source
    target_ds = targets[masked_source]
    first_channel = list(target_ds.data_vars.keys())[0]
    target = target_ds[first_channel].values
    pred_ds = preds[masked_source]
    pred = pred_ds[first_channel].values
    
    # Calculate shared min/max values
    vmin = min(target.min(), pred.min())
    vmax = max(target.max(), pred.max())

    # Plot masked source target
    ax = fig.add_subplot(gs[0, -2])
    dt = sample_df[sample_df["source_name"] == masked_source]["dt"].iloc[0]
    ax.imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)

    displayed_name = " ".join(masked_source.split("_")[3:5])
    displayed_name += " - " + first_channel
    title = f"{displayed_name}\n$\delta_t=${strfdelta(dt, '%H:%M:%S')}\n(MASKED)"
    ax.set_title(title)
    set_axis_ticks(ax, target_ds.lat.values, target_ds.lon.values)

    # Plot prediction
    ax = fig.add_subplot(gs[0, -1])
    title_suffix = "\nPrediction"

    ax.imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
    title = f"{displayed_name}\n$\delta_t=${strfdelta(dt, '%H:%M:%S')}{title_suffix}"
    ax.set_title(title)
    set_axis_ticks(ax, pred_ds.lat.values, pred_ds.lon.values)

    # Save the figure
    plt.tight_layout(h_pad=0.2, w_pad=0.4)  # Add specific padding parameters
    fig.savefig(results_dir / f"{batch_idx}_{sample_idx}.png", bbox_inches="tight")
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
