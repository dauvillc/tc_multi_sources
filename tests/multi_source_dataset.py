"""Tests the MultiSourceDataset class."""

import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange
from multi_sources.data_processing.multi_source_dataset import MultiSourceDataset
from multi_sources.data_processing.utils import read_variables_dict
from pyinstrument import Profiler


@hydra.main(config_path="../conf/", config_name="test", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    # Create the dataset
    dataset_dir = cfg["paths"]["preprocessed_dataset"]
    constants_dir = cfg["paths"]["constants"]
    included_vars = read_variables_dict(cfg['sources'])
    dataset = MultiSourceDataset(dataset_dir, constants_dir, included_vars, include_seasons=[2016])

    print(f"Dataset length: {len(dataset)} samples")
    # Try loading one sample
    sample = dataset[0]
    A, DT, CT, C, LM, D, V = sample[list(sample.keys())[0]]
    # Plot the data
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].imshow(C[0], cmap="gray")
    axs[0].set_title("Latitude")
    axs[1].imshow(D, cmap="gray")
    axs[1].set_title("Distance to center")
    axs[2].imshow(V[0], cmap="gray")
    axs[2].set_title("First variable")
    plt.savefig("tests/figures/multi_source_dataset_sample.png")

    # Profile an iteration over the dataset
    profiler = Profiler()
    profiler.start()
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)
    for i, batch in zip(trange(len(dataloader)), dataloader):
        pass
    profiler.stop()
    profiler.write_html("tests/outputs/profile_multi_source_dataset.html")
    profiler.print()


if __name__ == "__main__":
    main()
