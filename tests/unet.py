"""Tests the simple UNet in the MAE case"""
from multi_sources.models.unet import UNet
from multi_sources.structure.mae import MultisourceMaskedAutoencoder
from torch.utils.data import DataLoader
from multi_sources.data_processing.utils import read_source_file
from multi_sources.data_processing.multi_source_dataset import MultiSourceDataset


def main():
    # Load the sources
    sources = read_source_file()
    # Create the dataset
    dataset = MultiSourceDataset(sources)
    print(f"Dataset length: {len(dataset)} samples")
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    # Create the model
    model = UNet(dataset.get_n_variables(), 4, 2, 3).float()
    # Create the MAE
    mae = MultisourceMaskedAutoencoder(model)
    # Create a single batch and test the forward pass
    for batch in dataloader:
        break
    y = mae(batch)
    # Make sure the number of channels in the output is correct
    for source, preds in y.items():
        assert preds.shape[1] == dataset.get_n_variables()[source]
    return 0


if __name__ == "__main__":
    main()

