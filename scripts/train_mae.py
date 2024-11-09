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
from multi_sources.data_processing.collate_fn import multi_source_collate_fn


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("nan", lambda : float("nan"))
    cfg = OmegaConf.to_object(cfg)
    # If resume_run_id is in the config, load this run's cfg
    resume_run_id = cfg["resume_run_id"] if "resume_run_id" in cfg else None
    if resume_run_id:
        exp_cfg, checkpoint_path = load_experiment_cfg_from_checkpoint(
            cfg["paths"]["checkpoints"], resume_run_id
        )
        # For fields that define the experiment, use the values from the checkpoint:
        # - For the dataset, everything except the dataset_dir should come from the checkpoint.
        cfg["model"] = exp_cfg["model"]
        for split, split_cfg in cfg['dataset'].items():
            for key in split_cfg.keys():
                if key != 'dataset_dir':
                    split_cfg[key] = exp_cfg['dataset'][split][key]
        # The lightning module parameters should come from the checkpoint
        cfg["lightning_module"] = exp_cfg["lightning_module"]
        # The user can change fields from the checkpoint by setting them under the "change"
        # key (e.g. +change.lightning_module.masking_ratio=0.75)
        if "change" in cfg:
            changed_cfg = cfg.pop("change")
            cfg = update(cfg, changed_cfg)
    # Seed everything
    pl.seed_everything(cfg["seed"], workers=True)
    # Initialize Wandb and log the configuration
    if resume_run_id:
        wandb.init(**cfg["wandb"], config=cfg, dir=cfg["paths"]["wandb_logs"], resume="allow",
                   id=resume_run_id)
    else:
        wandb.init(**cfg["wandb"], config=cfg, dir=cfg["paths"]["wandb_logs"])
    # Create the logs directory if it does not exist
    Path(cfg["paths"]["wandb_logs"]).mkdir(parents=True, exist_ok=True)

    # Create the training dataset and dataloader
    train_dataset = hydra.utils.instantiate(cfg["dataset"]["train"])
    train_dataloader = DataLoader(train_dataset, **cfg["dataloader"], shuffle=True,
                                  collate_fn=multi_source_collate_fn)
    # Create the validation dataset and dataloader
    val_dataset = hydra.utils.instantiate(cfg["dataset"]["val"])
    val_dataloader = DataLoader(val_dataset, **cfg["dataloader"], shuffle=False,
                                collate_fn=multi_source_collate_fn)
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    source_names = train_dataset.get_source_names()
    context_vars = train_dataset.get_source_types_context_vars()
    # Create the backbone
    backbone = instantiate(cfg["model"]["backbone"])
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
            backbone=backbone,
            cfg=cfg,
            output_convs=output_convs,
            context_variables=context_vars,
        )
    else:
        pl_module = instantiate(
            cfg["lightning_module"],
            source_names,
            backbone,
            cfg,
            output_convs=output_convs,
            context_variables=context_vars,
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
    if resume_run_id:
        trainer.fit(pl_module, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(pl_module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
