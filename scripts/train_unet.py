"""Trains a UNet model on the multi-task autoencoding task."""

import pytorch_lightning as pl
import hydra
import wandb
from pathlib import Path
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
    # Initialize Wandb and log the configuration
    wandb.init(**cfg['wandb'], config=cfg, dir=cfg["paths"]["wandb_logs"])
    # Create the dataset
    metadata_path = cfg["paths"]["metadata"]
    dataset_dir = cfg["paths"]["preprocessed_dataset"]
    sources = read_sources(cfg["sources"])
    # Create the training dataset and dataloader
    train_dataset = MultiSourceDataset(
        metadata_path, dataset_dir, sources, include_seasons=cfg["experiment"]["train_seasons"],
        **cfg['dataset']
    )
    train_dataloader = DataLoader(train_dataset, **cfg["dataloader"])
    # Create the validation dataset and dataloader
    val_dataset = MultiSourceDataset(
        metadata_path, dataset_dir, sources, include_seasons=cfg["experiment"]["val_seasons"],
        **cfg['dataset']
    )
    val_dataloader = DataLoader(val_dataset, **cfg["dataloader"])
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    # Create the model
    model = UNet(train_dataset.get_n_variables(), 5, 64, 3).float()
    # Create the MAE
    mae = MultisourceMaskedAutoencoder(model)
    # Create the logger
    logger = WandbLogger(dir=cfg["paths"]["wandb_logs"])
    # Log the configuration
    logger.log_hyperparams(cfg)
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=Path(cfg["paths"]["checkpoints"]) / wandb.run.id,
        filename="unet-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    # Create the trainer
    trainer = pl.Trainer(
        logger=logger, log_every_n_steps=5, callbacks=[checkpoint_callback], **cfg["trainer"]
    )
    # Train the model
    trainer.fit(mae, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
