import os
from pathlib import Path
from time import localtime, strftime

import hydra
import lightning.pytorch as pl
import submitit
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from multi_sources.data_processing.collate_fn import multi_source_collate_fn
from multi_sources.data_processing.writer import MultiSourceWriter
from utils.cfg_utils import update
from utils.checkpoints import load_experiment_cfg_from_checkpoint


class PredictJob(submitit.helpers.Checkpointable):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self):
        cfg = self.cfg
        run_id = cfg["run_id"]

        # Load the experiment configuration from the checkpoint
        exp_cfg, checkpoint_path = load_experiment_cfg_from_checkpoint(
            cfg["paths"]["checkpoints"],
            run_id,
            best_or_latest="best",
        )
        # Update the experiment configuration with the current config.
        update(exp_cfg, cfg)

        # Create the dataset and dataloader
        split = cfg["split"]
        dataset = hydra.utils.instantiate(
            exp_cfg["dataset"][split],
        )
        # To ensure reproducibility, we seed a generator for the DataLoader specifically.
        dataloader_rng = torch.Generator().manual_seed(cfg["dataloader_seed"])
        dataloader = DataLoader(
            dataset,
            **exp_cfg["dataloader"],
            shuffle=cfg.get("shuffle_dataloader", False),
            collate_fn=multi_source_collate_fn,
            generator=dataloader_rng,
        )
        print("Dataset size:", len(dataset), f" ({split} split)")

        # Every set of predictions will be given a name.
        pred_name = cfg["pred_name"]
        # Create the results directory
        run_results_dir = Path(cfg["paths"]["predictions"]) / run_id / pred_name
        # Remove run_results_dir / info.csv if it exists
        if (run_results_dir / "info.csv").exists():
            (run_results_dir / "info.csv").unlink()

        # Instantiate the model and the lightning module
        pl_module = instantiate(
            exp_cfg["lightning_module"],
            dataset.sources,
            exp_cfg,
            validation_dir=None,
        )
        ckpt = torch.load(checkpoint_path, weights_only=False)
        pl_module.load_state_dict(ckpt["state_dict"])

        # Custom BasePredictionWriter to save the preds and targets with metadata (eg coords).
        writer = MultiSourceWriter(
            run_results_dir, dataset.dt_max, dataset=dataset, **cfg["writer"]
        )

        # Seed everything with the local seed, not the experiment's, to ensure
        # different models are evaluated with the same seed.
        pl.seed_everything(cfg["seed"], workers=True)

        trainer = pl.Trainer(
            **cfg["trainer"],
            callbacks=[writer],
            deterministic=True,
            logger=False,
            max_epochs=1,
        )
        trainer.predict(pl_module, dataloader, return_predictions=False)


def _make_executor(cfg: DictConfig) -> submitit.AutoExecutor:
    # Where submitit writes logs/stdout/err and its internal state
    folder = Path("submitit") / (
        "pred_" + cfg["wandb"]["name"] + f"_{strftime('%Y%m%d_%H-%M-%S', localtime())}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    ex = submitit.AutoExecutor(folder=str(folder), slurm_max_num_timeout=1)
    ex.update_parameters(**cfg["setup"])
    return ex


@hydra.main(version_base=None, config_path="../conf", config_name="make_predictions")
def main(cfg: DictConfig):
    # Enable the full errors in Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("nan", lambda: float("nan"))
    cfg = OmegaConf.to_object(cfg)

    # Create the job object and submit it to the auto executor.
    job = PredictJob(cfg)

    if cfg.get("launch_without_submitit", False):
        return job()
    else:
        executor = _make_executor(cfg)
        job = executor.submit(job)
        print(f"Submitted job {job.job_id}")
        return 0


if __name__ == "__main__":
    main()
