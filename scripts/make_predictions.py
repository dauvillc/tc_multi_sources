import pytorch_lightning as pl
import hydra
from torch.utils.data import DataLoader
from pathlib import Path
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from multi_sources.data_processing.writer import MultiSourceWriter


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    if "run_id" not in cfg:
        raise ValueError("Usage: python make_predictions.py run_id=<wandb_run_id>")
    run_id = cfg["run_id"]

    # Load the checkpoint
    checkpoints_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    checkpoint_path = checkpoints_dir / "epoch=90.ckpt"
    lightning_module_class = get_class(cfg["lightning_module"]["_target_"])
    module = lightning_module_class.load_from_checkpoint(checkpoint_path)

    # Create the validation dataset and dataloader
    val_dataset = hydra.utils.instantiate(cfg["dataset"]["val"], _convert_="partial")
    val_dataloader = DataLoader(val_dataset, **cfg["dataloader"])
    print("Validation dataset size:", len(val_dataset))

    # Create the results directory
    run_results_dir = Path(cfg["paths"]["predictions"]) / run_id

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
