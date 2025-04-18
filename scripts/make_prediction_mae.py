import lightning.pytorch as pl
import hydra
import torch

from torch.utils.data import DataLoader
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from multi_sources.data_processing.writer import MultiSourceWriter
from multi_sources.data_processing.collate_fn import multi_source_collate_fn
from utils.checkpoints import load_experiment_cfg_from_checkpoint
from utils.utils import update


@hydra.main(version_base=None, config_path="../conf", config_name="make_predictions")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("nan", lambda: float("nan"))
    cfg = OmegaConf.to_object(cfg)
    run_id = cfg["run_id"]

    # Load the experiment configuration from the checkpoint
    exp_cfg, checkpoint_path = load_experiment_cfg_from_checkpoint(
        cfg["paths"]["checkpoints"], run_id
    )
    # For some fields, we'll use the values from the config file instead of the ones from the
    # experiment
    exp_cfg["dataloader"].update(cfg["dataloader"])
    exp_cfg["paths"].update(cfg["paths"])
    split = "val"
    if "split" in cfg and cfg["split"] is not None:
        split = cfg["split"]
    exp_cfg["dataset"][split]["dataset_dir"] = cfg["paths"]["preprocessed_dataset"]

    # Finally, the user can use the "change" domain in the config to modify the parameters
    # of the experiment (e.g. +change.dataset.dt_max=xxx)
    if "change" in cfg:
        update(exp_cfg, cfg["change"])

    # Seed everything with the seed used in the experiment
    pl.seed_everything(exp_cfg["seed"], workers=True)

    # Create the dataset and dataloader
    dataset = hydra.utils.instantiate(
        exp_cfg["dataset"][split],
    )
    dataloader = DataLoader(
        dataset, **exp_cfg["dataloader"], shuffle=False, collate_fn=multi_source_collate_fn
    )
    print("Dataset size:", len(dataset), f" ({split} split)")

    # Create the results directory
    run_results_dir = Path(cfg["paths"]["predictions"]) / run_id
    # Remove run_results_dir / info.csv if it exists
    if (run_results_dir / "info.csv").exists():
        (run_results_dir / "info.csv").unlink()

    # Instantiate the model and the lightning module
    pl_module = instantiate(
        exp_cfg["lightning_module"],
        dataset.sources,
        exp_cfg,
        validation_dir=None
    )
    ckpt = torch.load(checkpoint_path)
    pl_module.load_state_dict(ckpt["state_dict"])

    # Custom BasePredictionWriter to save the preds and targets with metadata (eg coords).
    writer = MultiSourceWriter(run_results_dir, dataset.dt_max, dataset=dataset)

    trainer = pl.Trainer(
        **cfg["trainer"],
        callbacks=[writer],
        deterministic=True,
        logger=False,
        max_epochs=1,
    )
    trainer.predict(pl_module, dataloader)


if __name__ == "__main__":
    main()
