"""Tests the MultiSourceDataset class."""

import warnings
import matplotlib.pyplot as plt
import hydra
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange
from multi_sources.data_processing.multi_source_dataset import MultiSourceDataset
from multi_sources.data_processing.utils import read_variables_dict
from multi_sources.data_processing.collate_fn import multi_source_collate_fn
from pyinstrument import Profiler


@hydra.main(config_path="../conf/", config_name="test", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    # Create the dataset
    dataset_dir = cfg["paths"]["preprocessed_dataset"]
    split = "train"
    included_vars = read_variables_dict(cfg["sources"])
    dataset = MultiSourceDataset(dataset_dir, split, included_vars)

    print(f"Dataset length: {len(dataset)} samples")
    # Try loading one sample
    sample = dataset[0]
    first_source_sample = sample[list(sample.keys())[0]]
    C = first_source_sample["coords"]
    D = first_source_sample["dist_to_center"]
    V = first_source_sample["values"]
    # Plot the data
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].imshow(C[0], cmap="gray")
    axs[0].set_title("Latitude")
    axs[1].imshow(D, cmap="gray")
    axs[1].set_title("Distance to center")
    axs[2].imshow(V[0], cmap="gray")
    axs[2].set_title("First variable")
    plt.savefig("tests/figures/multi_source_dataset_sample.png")

    # Profile an iteration over the dataset, and compute the mean and std
    # of the samples in the dataset.
    profiler = Profiler()
    profiler.start()
    dataloader = DataLoader(
        dataset, batch_size=32, num_workers=cfg["num_workers"], shuffle=True,
        collate_fn=multi_source_collate_fn
    )

    data_means, data_stds = defaultdict(int), defaultdict(int)
    context_means, context_stds = defaultdict(int), defaultdict(int)
    with warnings.catch_warnings(action="ignore"):
        for i, batch in zip(trange(len(dataloader)), dataloader):
            for source_name, data in batch.items():
                # values
                v = data["values"].numpy()
                if np.isnan(v).all():
                    continue
                data_means[source_name] = data_means[source_name] + np.nanmean(v)
                mean_std = np.nanmean(np.nanstd(v, axis=(2, 3)))
                data_stds[source_name] = data_stds[source_name] + mean_std
                # context
                ct = data["context"]
                ct_nonan = ct[~ct.isnan()]
                context_means[source_name] = (
                    context_means[source_name] + ct_nonan.mean().item()
                )
                context_stds[source_name] = (
                    context_stds[source_name] + ct_nonan.std().item()
                )

    profiler.stop()
    profiler.write_html("tests/outputs/profile_multi_source_dataset.html")
    profiler.print()

    for source_name in data_means.keys():
        data_means[source_name] = data_means[source_name] / len(dataloader)
        data_stds[source_name] = data_stds[source_name] / len(dataloader)
        context_means[source_name] = context_means[source_name] / len(dataloader)
        context_stds[source_name] = context_stds[source_name] / len(dataloader)
        print(f"Source {source_name}:")
        print(
            f"Data mean: {data_means[source_name]}, Data std: {data_stds[source_name]}"
        )
        print(
            f"Context mean: {context_means[source_name]}, Context std: {context_stds[source_name]}"
        )


if __name__ == "__main__":
    main()
