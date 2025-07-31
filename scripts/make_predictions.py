import os
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from multi_sources.data_processing.collate_fn import multi_source_collate_fn
from multi_sources.data_processing.writer import MultiSourceWriter
from utils.cfg_utils import update
from utils.checkpoints import load_experiment_cfg_from_checkpoint


@hydra.main(version_base=None, config_path="../conf", config_name="make_predictions")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("nan", lambda: float("nan"))
    cfg = OmegaConf.to_object(cfg)
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
    dataloader = DataLoader(
        dataset, **exp_cfg["dataloader"], shuffle=False, collate_fn=multi_source_collate_fn
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
        masking_selection_seed=cfg["seed"],
    )
    ckpt = torch.load(checkpoint_path, weights_only=False)
    pl_module.load_state_dict(ckpt["state_dict"])

    # Custom BasePredictionWriter to save the preds and targets with metadata (eg coords).
    writer = MultiSourceWriter(run_results_dir, dataset.dt_max, dataset=dataset, **cfg["writer"])

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
    trainer.predict(pl_module, dataloader)


if __name__ == "__main__":
    # Enable the full errors in Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
