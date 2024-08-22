import lightning.pytorch as pl
import hydra
import wandb
import multi_sources
from hydra.utils import get_class, instantiate
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from utils.checkpoints import load_experiment_cfg_from_checkpoint
from utils.utils import update


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    # If resume_run_id is in the config, load this run's cfg
    resume_run_id = cfg["resume_run_id"] if "resume_run_id" in cfg else None
    if resume_run_id:
        exp_cfg, checkpoint_path = load_experiment_cfg_from_checkpoint(
            cfg["paths"]["checkpoints"], resume_run_id
        )
        # For fields that define the experiment, use the values from the checkpoint
        cfg["model"] = exp_cfg["model"]
        cfg["dataset"] = exp_cfg["dataset"]
        cfg["lightning_module"] = exp_cfg["lightning_module"]
        # The user can change fields from the checkpoint by setting them under the "change"
        # key (e.g. +change.lightning_module.masking_ratio=0.75)
        if "change" in cfg:
            changed_cfg = cfg.pop("change")
            cfg = update(cfg, changed_cfg)
    # Seed everything
    pl.seed_everything(cfg["seed"], workers=True)
    # Initialize Wandb and log the configuration
    wandb.init(**cfg["wandb"], config=cfg, dir=cfg["paths"]["wandb_logs"])
    # Create the logs directory if it does not exist
    Path(cfg["paths"]["wandb_logs"]).mkdir(parents=True, exist_ok=True)

    # Create the training dataset and dataloader
    train_dataset = hydra.utils.instantiate(cfg["dataset"]["train"])
    train_dataloader = DataLoader(train_dataset, **cfg["dataloader"], shuffle=True)
    # Create the validation dataset and dataloader
    val_dataset = hydra.utils.instantiate(cfg["dataset"]["val"])
    val_dataloader = DataLoader(val_dataset, **cfg["dataloader"])
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    source_names = train_dataset.get_source_names()
    # Create the encoder and decoder
    encoder = instantiate(cfg["model"]["encoder"])
    decoder = instantiate(cfg["model"]["decoder"])
    output_convs = None
    if "output_conv" in cfg["model"] and cfg["model"]["output_conv"]:
        # Instantiate one output conv per source
        output_convs = {
            source: instantiate(cfg["model"]["output_conv"]) for source in source_names
        }
    # Create the lightning module
    if resume_run_id:
        lightning_module_class = get_class(cfg["lightning_module"]["_target_"])
        pl_module = lightning_module_class.load_from_checkpoint(
            checkpoint_path,
            source_names=source_names,
            encoder=encoder,
            decoder=decoder,
            cfg=cfg,
            output_convs=output_convs,
        )
    else:
        pl_module = instantiate(
            cfg["lightning_module"], source_names, encoder, decoder, cfg, output_convs=output_convs
        )

    # Create the logger
    logger = WandbLogger(dir=cfg["paths"]["wandb_logs"], log_model=False)
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
