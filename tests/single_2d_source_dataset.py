"""Tests the Single2DSourceDataset class."""
import matplotlib.pyplot as plt
from multi_sources.data_processing.utils import read_source_file
from multi_sources.data_processing.single_2d_source_dataset import Single2DSourceDataset


def main():
    # Load the sources
    sources = read_source_file()
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
        plt.imshow(sample[var])
        plt.title(var)
    plt.savefig('tests/figures/single_2d_source_dataset.png')


if __name__ == "__main__":
    main()
