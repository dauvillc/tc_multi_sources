"""Trains a UNet model on the multi-task autoencoding task."""
import pytorch_lightning as pl
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
    dataloader = DataLoader(dataset, batch_size=64)
    # Create the model
    model = UNet(dataset.get_n_variables(), 4, 8, 3).float()
    # Create the MAE
    mae = MultisourceMaskedAutoencoder(model)
    # Create the trainer
    trainer = pl.Trainer(max_epochs=1)
    # Train the model
    trainer.fit(mae, dataloader)


if __name__ == "__main__":
    main()

