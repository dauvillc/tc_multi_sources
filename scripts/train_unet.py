"""Trains a UNet model on the multi-task autoencoding task."""

import pytorch_lightning as pl
import hydra
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from multi_sources.models.unet import UNet
from multi_sources.structure.mae import MultisourceMaskedAutoencoder
from torch.utils.data import DataLoader
from multi_sources.data_processing.multi_source_dataset import MultiSourceDataset
from multi_sources.data_processing.utils import read_sources
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    # Create the dataset
    metadata_path = cfg['paths']['metadata']
    dataset_dir = cfg['paths']['preprocessed_dataset']
    sources = read_sources(cfg['sources'])
    dataset = MultiSourceDataset(metadata_path, dataset_dir, sources,
                                 include_seasons=[2016])
    print(f"Dataset length: {len(dataset)} samples")
    # Create the dataloader
    dataloader = DataLoader(dataset, **cfg['dataloader'])
    # Create the model
    model = UNet(dataset.get_n_variables(), 5, 64, 3).float()
    # Create the MAE
    mae = MultisourceMaskedAutoencoder(model)
    # Create the logger
    logger = WandbLogger(project="multi-sources", name="test", dir=cfg["paths"]["wandb_logs"])
    # Log the configuration
    logger.log_hyperparams(cfg)
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg["paths"]["checkpoints"],
        filename='unet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    # Create the trainer
    trainer = pl.Trainer(logger=logger, log_every_n_steps=5,
                         callbacks=[checkpoint_callback],
                         **cfg['trainer'])
    # Train the model
    trainer.fit(mae, dataloader)


if __name__ == "__main__":
    main()
