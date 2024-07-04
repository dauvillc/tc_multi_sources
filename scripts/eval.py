"""Usage: python scripts/eval.py +run_id=<wandb_run_id>
Runs the evaluation on the validation or test set for a given run_id.
The predictions from that run must have been previously saved using
scripts/make_predictions.py.

# The predictions are saved in the following format:
# targets:
# - root_dir / targets / source_name / <batch_index.npy>
# predictions:
# - root_dir / outputs / source_name / <batch_index.npy>
# info dataframe:
# - root_dir / info.csv
# Each batch is an array of shape (batch_size, channels, height, width).
# the info dataframe contains the following columns:
# source_name (str), batch_idx (int), dt (float), available (float), masked (bool)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import hydra
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ProcessPoolExecutor


def process_batch(batch_idx, batch_df, results_dir, targets_dir, outputs_dir, source_names):
    """Processes a single batch, by displaying the inputs, target and prediction
    on a single figure for each element in the batch."""
    # Load the batch for every source
    targets = {
        source_name: np.load(targets_dir / source_name / f"{batch_idx}.npy")
        for source_name in source_names
    }
    outputs = {
        source_name: np.load(outputs_dir / source_name / f"{batch_idx}.npy")
        for source_name in source_names
    }
    n_sources = len(source_names)
    n_elements = len(targets[source_names[0]])
    # For each element in the batch and each source, retrieve whether the source was masked
    masked = {
        source_name: batch_df[batch_df.source_name == source_name].masked.values
        for source_name in source_names
    }
    # For each element in the batch, display it in an individual figure
    # such that:
    # - The number of columns is the number of sources if less than 5, otherwise 5
    # - The number of rows is sufficient to display all sources, plus one row for the prediction
    n_cols = min(n_sources, 5)
    n_rows = (n_sources + n_cols - 1) // n_cols + 1
    for n in range(n_elements):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        fig.suptitle(f"Batch {batch_idx}, element {n}")
        # Display the inputs
        for i, source_name in enumerate(source_names):
            target = targets[source_name][n]
            # The target array has shape (2 + n_channels, height, width)
            # The first two channels are the latitude and longitude of each pixel
            # Plot the first channel of the target
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(target[2], cmap="gray")
            ax.set_title(source_name)
            # Use the latitude and longitude to set the axis ticks
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xticks([0, target.shape[2] - 1])
            ax.set_yticks([0, target.shape[1] - 1])
            ax.set_xticklabels([f"{target[1, 0, 0]:.2f}", f"{target[1, 0, -1]:.2f}"])
            ax.set_yticklabels([f"{target[0, 0, 0]:.2f}", f"{target[0, -1, 0]:.2f}"])

        # Retrieve which source was masked
        masked_source = [source_name for source_name in source_names if masked[source_name][n]][0]
        # Display the prediction as the only subplot in the last row
        ax = axes[-1, 0]
        ax.imshow(outputs[masked_source][n, 0], cmap="gray")
        ax.set_title(f"Prediction - {masked_source}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xticks([0, outputs[masked_source].shape[3] - 1])
        ax.set_yticks([0, outputs[masked_source].shape[2] - 1])
        ax.set_xticklabels([f"{target[1, 0, 0]:.2f}", f"{target[1, 0, -1]:.2f}"])
        ax.set_yticklabels([f"{target[0, 0, 0]:.2f}", f"{target[0, -1, 0]:.2f}"])
        plt.savefig(results_dir / f"{batch_idx}_{n}.png")
        plt.close()


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    if "run_id" not in cfg:
        raise ValueError("Usage: python scripts/eval.py run_id=<wandb_run_id>")
    run_id = cfg["run_id"]

    root_dir = Path(cfg["paths"]["predictions"]) / run_id
    if not root_dir.exists():
        raise ValueError(
            f"Predictions for run_id {run_id} do not exist.\
                Please run scripts/make_predictions.py first."
        )
    targets_dir = root_dir / "targets"
    outputs_dir = root_dir / "outputs"
    info_filepath = root_dir / "info.csv"
    # Fetch the list of source names
    source_names = [source_name.name for source_name in targets_dir.iterdir()]

    results_dir = Path(cfg["paths"]["results"]) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    num_workers = cfg["num_workers"]
    n_displayed_batches = cfg["n_displayed_batches"]

    # ==== VISUAL EVALUATION ====
    # Load the info dataframe
    info_df = pd.read_csv(info_filepath)
    # Keep only the first n_displayed_batches batches
    info_df = info_df[info_df.batch_idx < n_displayed_batches]
    # Process the batch sequentially
    for batch in range(n_displayed_batches):
        batch_df = info_df[info_df.batch_idx == batch]
        process_batch(batch, batch_df, results_dir, targets_dir, outputs_dir, source_names)


if __name__ == "__main__":
    main()
