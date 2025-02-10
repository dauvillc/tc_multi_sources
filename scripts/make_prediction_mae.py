import lightning.pytorch as pl
import hydra
from torch.utils.data import DataLoader
from pathlib import Path
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from multi_sources.data_processing.writer_mae import MultiSourceWriter
from multi_sources.data_processing.collate_fn import multi_source_collate_fn
from multi_sources.models.recorder import AttentionRecorder
from utils.checkpoints import load_experiment_cfg_from_checkpoint


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
    if split == "test" and "test" not in exp_cfg["dataset"]:
        exp_cfg["dataset"]["test"] = exp_cfg["dataset"]["val"]
        exp_cfg['dataset']["test"]["split"] = "test"
    exp_cfg["dataset"][split]["dataset_dir"] = cfg["paths"]["preprocessed_dataset"]
    # Seed everything with the seed used in the experiment
    pl.seed_everything(exp_cfg["seed"], workers=True)

    # Create the validation dataset and dataloader
    val_dataset = hydra.utils.instantiate(
        exp_cfg["dataset"][split],
    )
    val_dataloader = DataLoader(
        val_dataset, **exp_cfg["dataloader"], shuffle=False, collate_fn=multi_source_collate_fn
    )
    print("Validation dataset size:", len(val_dataset))

    # Create the results directory
    run_results_dir = Path(cfg["paths"]["predictions"]) / run_id
    # Remove run_results_dir / info.csv if it exists
    if (run_results_dir / "info.csv").exists():
        (run_results_dir / "info.csv").unlink()

    # Instantiate the model and the lightning module
    backbone = instantiate(exp_cfg["model"]["backbone"])

    lightning_module_class = get_class(exp_cfg["lightning_module"]["_target_"])
    module = lightning_module_class.load_from_checkpoint(
        checkpoint_path,
        sources=val_dataset.sources,
        backbone=backbone,
        cfg=exp_cfg,
    )

    # If we are saving the attention maps, wrap the decoder with the AttentionRecorder class
    if cfg["save_attention_maps"]:
        maps_dir = run_results_dir / "attention_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        module.backbone = AttentionRecorder(module.backbone, maps_dir)

    # Make predictions using the MultiSourceWriter class, which is a custom implementation of
    # BasePredictionWriter.
    module.eval()
    writer = MultiSourceWriter(run_results_dir, val_dataset.dt_max, dataset=val_dataset)
    trainer = pl.Trainer(
        **cfg["trainer"],
        callbacks=[writer],
    )
    trainer.predict(module, val_dataloader)


if __name__ == "__main__":
    main()
