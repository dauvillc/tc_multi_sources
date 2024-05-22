"""Tests the MultiSourceDataset class."""
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange
from multi_sources.data_processing.multi_source_dataset import MultiSourceDataset
from multi_sources.data_processing.utils import read_sources
from pyinstrument import Profiler


@hydra.main(config_path="../conf/", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    # Create the dataset
    metadata_path = cfg['paths']['metadata']
    dataset_dir = cfg['paths']['preprocessed_dataset']
    sources = read_sources(cfg['sources'])
    dataset = MultiSourceDataset(metadata_path, dataset_dir, sources,
                                 include_seasons=[2016])
    
    print(f"Dataset length: {len(dataset)} samples")
    # Try loading one sample
    sample = dataset[0]
    S, DT, C, D, V = sample[list(sample.keys())[0]]
    # Plot the data
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].imshow(C[0], cmap="gray")
    axs[0].set_title("Latitude")
    axs[1].imshow(D, cmap="gray")
    axs[1].set_title("Distance to center")
    axs[2].imshow(V[0], cmap="gray")
    axs[2].set_title("First variable")
    plt.savefig('tests/figures/multi_source_dataset_sample.png')

    # Profile an iteration over the dataset
    profiler = Profiler()
    profiler.start()
    dataloader = DataLoader(dataset, batch_size=64, num_workers=2, shuffle=True)
    for i, batch in zip(trange(len(dataloader)), dataloader):
        pass
    profiler.stop()
    profiler.open_in_browser()


if __name__ == "__main__":
    main()
