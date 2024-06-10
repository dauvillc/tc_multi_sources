import lightning.pytorch as pl
import hydra
from torch.utils.data import DataLoader
from pathlib import Path
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from multi_sources.data_processing.writer import MultiSourceWriter


@hydra.main(version_base=None, config_path="../conf", config_name="make_predictions")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    run_id = cfg["run_id"]

    # Create the validation dataset and dataloader
    val_dataset = hydra.utils.instantiate(cfg["dataset"]["val"], _convert_="partial")
    val_dataloader = DataLoader(val_dataset, **cfg["dataloader"])
    print("Validation dataset size:", len(val_dataset))

    # Load the checkpoint. The checkpoints are stored as "epoch=xx.ckpt" files in the checkpoints
    # directory.
    # Use the latest checkpoint.
    checkpoints_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
    checkpoint_path = checkpoint_files[-1]
    print("Loading checkpoint:", checkpoint_path.stem)
    lightning_module_class = get_class(cfg["lightning_module"]["_target_"])
    module = lightning_module_class.load_from_checkpoint(checkpoint_path)

    # Create the results directory
    run_results_dir = Path(cfg["paths"]["predictions"]) / run_id
    # Remove run_results_dir / info.csv if it exists
    if (run_results_dir / "info.csv").exists():
        (run_results_dir / "info.csv").unlink()

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
