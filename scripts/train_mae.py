import lightning.pytorch as pl
import hydra
import multi_sources
from hydra.utils import get_class, instantiate
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from utils.checkpoints import load_experiment_cfg_from_checkpoint
from utils.utils import update, get_random_code
from multi_sources.data_processing.collate_fn import multi_source_collate_fn


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("nan", lambda: float("nan"))
    cfg = OmegaConf.to_object(cfg)
    # If resume_run_id is in the config, load this run's cfg
    resume_run_id = cfg["resume_run_id"] if "resume_run_id" in cfg else None
    # Create a random id for the run if it is not resuming
    run_id = get_random_code() if not resume_run_id else resume_run_id
    if resume_run_id:
        exp_cfg, checkpoint_path = load_experiment_cfg_from_checkpoint(
            cfg["paths"]["checkpoints"], resume_run_id
        )
        # For fields that define the experiment, use the values from the checkpoint:
        # - For the dataset, everything except the dataset_dir should come from the checkpoint.
        cfg["model"] = exp_cfg["model"]
        for split, split_cfg in cfg["dataset"].items():
            for key in split_cfg.keys():
                if key != "dataset_dir":
                    split_cfg[key] = exp_cfg["dataset"][split][key]
        # The lightning module parameters should come from the checkpoint
        cfg["lightning_module"] = exp_cfg["lightning_module"]
        # The user can change fields from the checkpoint by setting them under the "change"
        # key (e.g. +change.lightning_module.masking_ratio=0.75)
        if "change" in cfg:
            changed_cfg = cfg.pop("change")
            cfg = update(cfg, changed_cfg)
    # Seed everything
    pl.seed_everything(cfg["seed"], workers=True)

    # Create the training dataset and dataloader
    train_dataset = hydra.utils.instantiate(cfg["dataset"]["train"])
    train_dataloader = DataLoader(
        train_dataset, **cfg["dataloader"], shuffle=True, collate_fn=multi_source_collate_fn,
        drop_last=True,
    )
    # Create the validation dataset and dataloader
    val_dataset = hydra.utils.instantiate(cfg["dataset"]["val"])
    val_dataloader = DataLoader(
        val_dataset,
        **cfg["dataloader"],
        shuffle=False,
        collate_fn=multi_source_collate_fn,
        drop_last=True,
    )
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    # Create the backbone
    backbone = instantiate(cfg["model"]["backbone"])
    # Create the lightning module
    if resume_run_id:
        lightning_module_class = get_class(cfg["lightning_module"]["_target_"])
        pl_module = lightning_module_class.load_from_checkpoint(
            checkpoint_path,
            sources=train_dataset.sources,
            backbone=backbone,
            cfg=cfg,
        )
    else:
        pl_module = instantiate(
            cfg["lightning_module"],
            train_dataset.sources,
            backbone,
            cfg,
        )

    # Create the logs directory if it does not exist
    Path(cfg["paths"]["wandb_logs"]).mkdir(parents=True, exist_ok=True)
    # Create the logger
    if resume_run_id:
        logger = WandbLogger(
            **cfg["wandb"],
            dir=cfg["paths"]["wandb_logs"],
            save_dir=cfg["paths"]["wandb_logs"],
            log_model=False,
            config=cfg,
            id=resume_run_id,
            resume="allow",
        )
    else:
        logger = WandbLogger(
            **cfg["wandb"],
            log_model=False,
            config=cfg,
            id=run_id,
            dir=cfg["paths"]["wandb_logs"],
            save_dir=cfg["paths"]["wandb_logs"],
        )
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=Path(cfg["paths"]["checkpoints"]) / run_id,
        filename="{epoch:02d}",
        save_top_k=1,
        mode="min",
    )
    # Create the trainer
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, LearningRateMonitor()],
        **cfg["trainer"],
    )
    # Train the model
    if resume_run_id:
        trainer.fit(pl_module, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(pl_module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
