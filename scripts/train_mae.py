import lightning.pytorch as pl
import hydra
import multi_sources
import torch
from hydra.utils import get_class, instantiate
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from utils.utils import get_random_code
from utils.checkpoints import load_experiment_cfg_from_checkpoint
from multi_sources.data_processing.collate_fn import multi_source_collate_fn


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("nan", lambda: float("nan"))
    cfg = OmegaConf.to_object(cfg)

    resume_run_id = cfg["resume_run_id"] if "resume_run_id" in cfg else None
    # If a run is resuming, the resume_mode must be set to either 'resume' or 'fine_tune'
    if resume_run_id and (
        "resume_mode" not in cfg or cfg["resume_mode"] not in ["resume", "fine_tune"]
    ):
        raise ValueError(
            "If resume_run_id is set, resume_mode must be set to either"
            " 'resume' or 'fine_tune' with +resume_mode=..."
        )
    # Create a random id for the run
    run_id = get_random_code()
    if resume_run_id:
        run_id = resume_run_id + "-" + run_id
        _, checkpoint_path = load_experiment_cfg_from_checkpoint(
            cfg["paths"]["checkpoints"], resume_run_id
        )
    print("Run ID:", run_id)

    # Seed everything
    pl.seed_everything(cfg["seed"], workers=True)

    # Create the training dataset and dataloader
    train_dataset = hydra.utils.instantiate(cfg["dataset"]["train"])
    train_dataloader = DataLoader(
        train_dataset,
        **cfg["dataloader"],
        shuffle=True,
        collate_fn=multi_source_collate_fn,
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
    # Create the validation directory if it does not exist
    val_dir = Path(cfg["paths"]["validation"]) / run_id
    val_dir.mkdir(parents=True, exist_ok=True)

    # Create the lightning module
    pl_module = instantiate(
        cfg["lightning_module"],
        train_dataset.sources,
        cfg,
        validation_dir=val_dir,
    )
    if resume_run_id:
        # The following snippet loads the weights of the checkpoint into the new
        # lightning module, but allowing new weights that are not in the checkpoint.
        # It also allows weights that are in the checkpoint but not in the new lightning
        # module to be ignored.
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/39
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        former_dict = ckpt["state_dict"]
        # The user can add "+reset_output_layers=true" to the command line to reset the
        # output layers of the model. In this case, the output layers are not loaded from
        # the checkpoint.
        if "reset_output_layers" in cfg and cfg["reset_output_layers"]:
            former_dict = {k: v for k, v in former_dict.items() if "output_proj" not in k}
        new_dict = pl_module.state_dict()
        # add to the former dict the weights that were added in the new dict.
        for k, v in new_dict.items():
            if k not in former_dict:
                former_dict[k] = v
        # Remove from the former dict the weights that are not in the new dict.
        for k in list(former_dict.keys()):
            if k not in new_dict:
                del former_dict[k]
        # 3. load the former state dict that now contains all the weights.
        pl_module.load_state_dict(former_dict)

    # Callbacks
    # Create the logs directory if it does not exist
    Path(cfg["paths"]["wandb_logs"]).mkdir(parents=True, exist_ok=True)
    # Create the logger
    logger = WandbLogger(
        **cfg["wandb"],
        log_model=False,
        config=cfg,
        id=run_id,
        dir=cfg["paths"]["wandb_logs"],
        save_dir=cfg["paths"]["wandb_logs"],
    )

    # Model checkpoint after every epoch
    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg["paths"]["checkpoints"]) / run_id,
        filename=f"{run_id}-" + "{epoch}-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor()
    callbacks = [epoch_checkpoint_callback, lr_monitor]
                 
    # Create the trainer
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=500,
        callbacks=callbacks,
        **cfg["trainer"],
    )

    # Train the model
    if resume_run_id and cfg["resume_mode"] == "resume":
        trainer.fit(pl_module, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(pl_module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
