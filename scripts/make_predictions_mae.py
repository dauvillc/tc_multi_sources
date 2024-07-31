import lightning.pytorch as pl
import hydra
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from multi_sources.data_processing.writer_mae import MultiSourceWriter
from multi_sources.models.recorder import AttentionRecorder
from utils.checkpoints import load_experiment_cfg_from_checkpoint


@hydra.main(version_base=None, config_path="../conf", config_name="make_predictions")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    run_id = cfg["run_id"]

    # Load the experiment configuration from the checkpoint
    exp_cfg, checkpoint_path = load_experiment_cfg_from_checkpoint(
        cfg["paths"]["checkpoints"], run_id
    )
    # Seed everything with the seed used in the experiment
    pl.seed_everything(exp_cfg["seed"], workers=True)

    # Create the validation dataset and dataloader
    val_dataset = hydra.utils.instantiate(exp_cfg["dataset"]["val"], _convert_="partial")
    exp_cfg["dataloader"].update(cfg["dataloader"])
    val_dataloader = DataLoader(val_dataset, **exp_cfg["dataloader"])
    print("Validation dataset size:", len(val_dataset))

    # Create the results directory
    run_results_dir = Path(cfg["paths"]["predictions"]) / run_id
    # Remove run_results_dir / info.csv if it exists
    if (run_results_dir / "info.csv").exists():
        (run_results_dir / "info.csv").unlink()

    # Instantiate the model and the lightning module
    encoder = instantiate(exp_cfg["model"]["encoder"])
    decoder = instantiate(exp_cfg["model"]["decoder"])
    lightning_module_class = get_class(exp_cfg["lightning_module"]["_target_"])
    module = lightning_module_class.load_from_checkpoint(
        checkpoint_path, encoder=encoder, decoder=decoder, cfg=exp_cfg
    )

    # If we are saving the attention maps, wrap the decoder with the AttentionRecorder class
    if cfg["save_attention_maps"]:
        maps_dir = run_results_dir / "attention_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        module.decoder = AttentionRecorder(module.decoder, maps_dir)

    # Make predictions using the MultiSourceWriter class, which is a custom implementation of
    # BasePredictionWriter.
    module.eval()
    writer = MultiSourceWriter(run_results_dir)
    trainer = pl.Trainer(
        **cfg["trainer"],
        callbacks=[writer],
    )
    trainer.predict(module, val_dataloader)


if __name__ == "__main__":
    main()
