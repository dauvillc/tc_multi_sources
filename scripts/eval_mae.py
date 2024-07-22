"""Usage: python scripts/eval_mae.py +run_id=<wandb_run_id>
Runs the evaluation on the validation or test set for a given run_id.
The predictions from that run must have been previously saved using
scripts/make_predictions_mae.py.

# The predictions are saved in the following format:
# targets:
# - root_dir / targets / source_name / <batch_index.npy>
# predictions:
# - root_dir / outputs / source_name / <batch_index.npy>
# info dataframe:
# - root_dir / info.csv
# Each batch is an array of shape (batch_size, channels, height, width).
# the info dataframe contains the following columns:
# source_name (str), batch_idx (int), dt (float)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import hydra
import pandas as pd
from tqdm import trange, tqdm
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ProcessPoolExecutor


def process_batch(
    batch_idx, batch_df, results_dir, targets_dir, outputs_dir, attention_maps_dir, source_names
):
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
    # For each element in the batch, display it in an individual figure
    # such that:
    # - There are two columns, one for the prediction (left) and one for the target (right)
    # - There is one row per source
    n_rows = n_sources
    n_cols = 2
    for i in range(n_elements):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
        for j, source_name in enumerate(source_names):
            target = targets[source_name][i]
            # The target array has shape (3, height, width)
            # The first two channels are the latitude and longitude of each pixel
            coords = target[:2]
            target = target[2]
            # Plot the target
            ax = axes[j, 0]
            ax.imshow(target, cmap="gray")
            ax.set_title(f"{source_name} - target")
            ax.axis("off")
            # Set the ticks to the coordinates
            ax.set_xticks(np.arange(target.shape[1]))
            ax.set_xticklabels(coords[0, 0])
            ax.set_yticks(np.arange(target.shape[0]))
            ax.set_yticklabels(coords[1, :, 0])
            # Plot the prediction
            prediction = outputs[source_name][i][0]
            ax = axes[j, 1]
            ax.imshow(prediction, cmap="gray")
            ax.set_title(f"{source_name} - prediction")
            ax.axis("off")
            # Set the ticks to the coordinates
            ax.set_xticks(np.arange(prediction.shape[1]))
            ax.set_xticklabels(coords[0, 0])
            ax.set_yticks(np.arange(prediction.shape[0]))
            ax.set_yticklabels(coords[1, :, 0])

        plt.tight_layout()
        plt.savefig(results_dir / f"{batch_idx}_{i}.png")
        plt.close()

    # If results_dir/attention_maps exists, it contains the attention maps for every batch
    # as <batch_idx>.npy, of shape (B, head, n_tokens, n_tokens)
    if attention_maps_dir.exists():
        attention_maps = np.load(attention_maps_dir / f"{batch_idx}.npy")
        # Display the attention maps for the first element in the batch,
        # with at most 4 columns
        n_heads = attention_maps.shape[1]
        n_cols = min(n_heads, 4)
        n_rows = (n_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
        for i in range(n_heads):
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(attention_maps[0, i], cmap="viridis")
            ax.set_title(f"Head {i}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(results_dir / f"{batch_idx}_attention_maps.png")


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
    attention_maps_dir = root_dir / "attention_maps"
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
    if num_workers < 2:
        # Process the batch sequentially
        for batch in trange(n_displayed_batches):
            batch_df = info_df[info_df.batch_idx == batch]
            process_batch(
                batch,
                batch_df,
                results_dir,
                targets_dir,
                outputs_dir,
                attention_maps_dir,
                source_names,
            )
    else:
        # Process the batch in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_batch,
                    batch,
                    info_df[info_df.batch_idx == batch],
                    results_dir,
                    targets_dir,
                    outputs_dir,
                    attention_maps_dir,
                    source_names,
                )
                for batch in range(n_displayed_batches)
            ]
            for future in tqdm(futures, total=n_displayed_batches):
                future.result()


if __name__ == "__main__":
    main()
