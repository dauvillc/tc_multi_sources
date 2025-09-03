import os
from datetime import timedelta
from pathlib import Path
from time import localtime, strftime

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import submitit
from multi_sources.data_processing.collate_fn import multi_source_collate_fn
from utils.cfg_utils import get_random_code
from utils.checkpoints import load_experiment_cfg_from_checkpoint, load_weights_intersection


class TrainJob(submitit.helpers.Checkpointable):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self):
        cfg = self.cfg
        resume_run_id = cfg["resume_run_id"] if "resume_run_id" in cfg else None
        # If a run is resuming, the resume_mode option can be set to either "resume" or "fine_tune".
        # By default, it is set to "resume".
        if resume_run_id:
            if "resume_mode" not in cfg:
                cfg["resume_mode"] = "resume"
            elif cfg["resume_mode"] not in ["resume", "fine_tune"]:
                raise ValueError(
                    f"Invalid resume mode: {cfg['resume_mode']}. "
                    "It must be either 'resume' or 'fine_tune'."
                )

            print("Resuming run: ", resume_run_id)
            # Create the run's ID and retrieve its checkpoint path.
            # Since resuming W&B offline runs is unstable, we won't use exactly the same run ID.
            # Instead, we'll use run-n where n starts at 1 (first resume) and increments by 1
            # for each subsequent resume.
            split = resume_run_id.split("-")
            if len(split) > 1 and split[-1].isdigit():
                # If the run ID ends with a number, increment it.
                run_id = "-".join(split[:-1]) + "-" + str(int(split[-1]) + 1)
            else:
                # If the run ID does not end with a number, append "-1".
                run_id = resume_run_id + "-1"
            _, checkpoint_path = load_experiment_cfg_from_checkpoint(
                cfg["paths"]["checkpoints"], resume_run_id, best_or_latest="latest"
            )
        else:
            run_id = get_random_code()
        print("Run ID:", run_id)
        self.run_id = run_id

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
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            former_dict = ckpt["state_dict"]
            # The user can add "+reset_output_layers=true" to the command line to reset the
            # output layers of the model. In this case, the output layers are not loaded from
            # the checkpoint.
            if "reset_output_layers" in cfg and cfg["reset_output_layers"]:
                former_dict = {k: v for k, v in former_dict.items() if "output_proj" not in k}
            new_dict = pl_module.state_dict()
            # Only keep the intersection of the keys in the former and current dictionaries.
            former_dict = load_weights_intersection(former_dict, new_dict)
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

        print("Saving checkpoints to:", cfg["paths"]["checkpoints"])
        # Model checkpoint after every epoch if val_loss improves
        epoch_checkpoint_callback = ModelCheckpoint(
            dirpath=Path(cfg["paths"]["checkpoints"]) / run_id,
            filename=f"{run_id}-" + "{epoch}-best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        # Model checkpoint every 30 minutes
        time_checkpoint_callback = ModelCheckpoint(
            dirpath=Path(cfg["paths"]["checkpoints"]) / run_id,
            filename=f"{run_id}-" + "{epoch}-{step}",
            train_time_interval=timedelta(minutes=30),
            save_top_k=-1,  # Save all checkpoints
        )

        lr_monitor = LearningRateMonitor()
        model_summary = ModelSummary(max_depth=3)
        callbacks = [
            epoch_checkpoint_callback,
            time_checkpoint_callback,
            lr_monitor,
            model_summary,
        ]

        # Create the trainer
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=500,
            callbacks=callbacks,
            deterministic=True,
            **cfg["trainer"],
        )

        # Train the model
        if resume_run_id and cfg["resume_mode"] == "resume":
            trainer.fit(pl_module, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
        else:
            trainer.fit(pl_module, train_dataloader, val_dataloader)

    def checkpoint(self):
        """Called by submitit on SIGUSR1."""
        new_cfg = self.cfg.copy()
        new_cfg["resume_run_id"] = self.run_id
        new_cfg["resume_mode"] = "resume"
        return submitit.helpers.DelayedSubmission(TrainJob(new_cfg))


def _make_executor(cfg: DictConfig) -> submitit.AutoExecutor:
    # Where submitit writes logs/stdout/err and its internal state
    folder = Path("submitit") / (
        cfg["wandb"]["name"] + f"_{strftime('%Y%m%d_%H-%M-%S', localtime())}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    ex = submitit.AutoExecutor(folder=str(folder), slurm_max_num_timeout=20)
    ex.update_parameters(**cfg["setup"])
    return ex


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    # Enable the full errors in Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("nan", lambda: float("nan"))
    cfg = OmegaConf.to_object(cfg)

    # Create the job object and submit it to the auto executor.
    job = TrainJob(cfg)

    if cfg.get("launch_without_submitit", False):
        # If not using submitit, we run the job directly.
        return job()
    else:
        executor = _make_executor(cfg)
        job = executor.submit(job)
        print(f"Submitted job {job.job_id}")
        return 0


if __name__ == "__main__":
    main()
