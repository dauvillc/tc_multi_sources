"""Usage: python scripts/eval.py +run_id=<wandb_run_id>
Runs the evaluation on the validation or test set for a given run_id.
The predictions from that run must have been previously saved using
scripts/make_predictions.py.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import hydra
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ProcessPoolExecutor


def process_source_name(source_name, results_dir, targets_dir, outputs_dir, info_filepath):
    """Saves the images for a given source_name."""
    # The predictions are saved in the following format:
    # targets:
    # - root_dir / targets / source_name / <batch_index.npy>
    # predictions:
    # - root_dir / predictions / source_name / <batch_index.npy>
    # info dataframe:
    # - root_dir / info.csv
    # Each batch is an array of shape (batch_size, channels, height, width).
    # For each batch, split it into the individual images and plot them.
    # Save the images under
    # results_dir / source_name / <image_index>.png
    # Each image should have one row per channel, and two columns: target and prediction.
    # The info dataframe has the following columns:
    # source_name, batch_idx, dt, available, masked
    # On top of each image, write the following information:
    # dt, available, masked
    print(f"Processing {source_name}")
    source_dir = results_dir / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir = targets_dir / source_name
    prediction_dir = outputs_dir / source_name
    batch_indices = [int(batch_index.stem) for batch_index in target_dir.iterdir()]
    # Load the info dataframe
    info_df = pd.read_csv(info_filepath)
    for batch_index in batch_indices:
        target = np.load(target_dir / f"{batch_index}.npy")
        prediction = np.load(prediction_dir / f"{batch_index}.npy")
        # Isolate the batch in the info dataframe
        batch_info = info_df[info_df["batch_idx"] == batch_index]
        # Fill NaNs with zeros in the target to be coherent with the prediction
        target = np.nan_to_num(target)
        for i in range(target.shape[0]):
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(target[i, 0], cmap="gray")
            axs[0].set_title("Target")
            axs[0].axis("off")
            axs[1].imshow(prediction[i, 0], cmap="gray")
            axs[1].set_title("Prediction")
            axs[1].axis("off")
            # Write the information on top of the image
            dt = batch_info["dt"].values[i]
            available = batch_info["available"].values[i]
            masked = batch_info["masked"].values[i]
            fig.suptitle(f"dt: {dt}, available: {available}, masked: {masked}")
            fig.savefig(source_dir / f"{batch_index}_{i}.png")
            plt.close(fig)


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

    # ==== VISUAL EVALUATION ====
    # Process the source names in parallel.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for source_name in source_names:
            futures.append(
                executor.submit(
                    process_source_name, source_name, results_dir, targets_dir, outputs_dir,
                    info_filepath
                )
            )
        for future in tqdm(futures):
            future.result()


if __name__ == "__main__":
    main()
