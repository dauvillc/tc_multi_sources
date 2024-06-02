import pytorch_lightning as pl
import hydra
import wandb
import multi_sources
from hydra.utils import instantiate
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    # Initialize Wandb and log the configuration
    wandb.init(**cfg["wandb"], config=cfg, dir=cfg["paths"]["wandb_logs"])
    # Create the logs directory if it does not exist
    Path(cfg["paths"]["wandb_logs"]).mkdir(parents=True, exist_ok=True)

    # Create the training dataset and dataloader
    train_dataset = hydra.utils.instantiate(cfg["dataset"]["train"], _convert_="partial")
    train_dataloader = DataLoader(train_dataset, **cfg["dataloader"], shuffle=True)
    # Create the validation dataset and dataloader
    val_dataset = hydra.utils.instantiate(cfg["dataset"]["val"], _convert_="partial")
    val_dataloader = DataLoader(val_dataset, **cfg["dataloader"])
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    # Create the model
    model = instantiate(cfg["model"], train_dataset.get_n_variables()).float()
    # Create the lightning module
    pl_module = instantiate(cfg["lightning_module"], model)

    # Create the logger
    logger = WandbLogger(dir=cfg["paths"]["wandb_logs"], log_model='all')
    # Log the configuration
    logger.log_hyperparams(cfg)
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=Path(cfg["paths"]["checkpoints"]) / wandb.run.id,
        filename="{epoch:02d}",
        save_top_k=1,
        mode="min",
    )
    # Create the trainer
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, LearningRateMonitor()],
        **cfg["trainer"]
    )
    # Train the model
    trainer.fit(pl_module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
