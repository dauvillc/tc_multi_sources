"""Tests the MultiSourceDataset class."""
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from time import time
from tqdm import trange
from multi_sources.data_processing.utils import read_source_file
from multi_sources.data_processing.multi_source_dataset import MultiSourceDataset


def main():
    # Load the sources
    sources = read_source_file()
    # Create the dataset
    dataset = MultiSourceDataset(sources)
    print(f"Dataset length: {len(dataset)} samples")
    # Try loading one sample
    sample = dataset[0]
    S, DT, C, D, V = sample[dataset.sources[0].name]
    # Plot the data
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].imshow(C[0], cmap="gray")
    axs[0].set_title("Latitude")
    axs[1].imshow(D, cmap="gray")
    axs[1].set_title("Distance to center")
    axs[2].imshow(V[0], cmap="gray")
    axs[2].set_title("First variable")
    plt.savefig('tests/figures/multi_source_dataset_sample.png')

    # Measure the time to iterate over the dataset with a DataLoader
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True)
    start = time()
    for i, batch in zip(trange(len(dataset) // 64), dataloader):
        pass
    print(f"Time to iterate over all samples: {time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
