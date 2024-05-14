"""Tests the Single2DSourceDataset class."""
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from time import time
from tqdm import trange
from multi_sources.data_processing.utils import read_sources
from multi_sources.data_processing.single_2d_source_dataset import Single2DSourceDataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    # Load the sources
    sources = read_sources(cfg['experiment']['sources'])
    # Create the dataset from the first source
    dataset = Single2DSourceDataset(sources[0])
    print(f"Dataset length: {len(dataset)} samples")
    # Try loading one sample
    sample = dataset[0]
    # Get the list of variables from that source
    variables = dataset.source.variables
    # Plot each variable
    for i, var in enumerate(variables):
        plt.subplot(1, len(variables), i + 1)
        plt.imshow(sample[i])
        plt.title(var)
    plt.savefig('tests/figures/single_2d_source_dataset.png')

    # Now, create a DataLoader from the dataset, and measure the time needed
    # to iterate over all samples
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    start = time()
    for i, batch in zip(trange(len(dataset) // 64), dataloader):
        pass
    print(f"Time to iterate over all samples: {time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
