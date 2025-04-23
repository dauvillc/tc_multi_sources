"""Implements small utility functions for loading checkpoints."""

from pathlib import Path

import torch


def load_experiment_cfg_from_checkpoint(checkpoints_dir, run_id):
    """Loads a Lightning checkpoint and extracts from it the
    experiment configuration (which must have been saved as the "cfg"
    hyperparameter of the LightningModule).

    Args:
        checkpoints_dir (str): The directory where the checkpoints are stored.
        run_id (str): The ID of the run.
    Returns:
        exp_cfg (dict): The experiment configuration.
        checkpoint_path (str): The path to the checkpoint.
    """
    # Load the checkpoint. The checkpoints are stored as "epoch=xx.ckpt" files in the checkpoints
    # directory.
    # Use the latest checkpoint.
    checkpoints_dir = Path(checkpoints_dir) / run_id
    checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
    checkpoint_path = checkpoint_files[-1]
    print("Loading checkpoint:", checkpoint_path.stem, " from run ", run_id)
    # The checkpoint includes the whole configuration dict of the experiment, in
    # checkpoint["hyper_parameters"]['cfg']. We'll use this to reproduce
    # the exact configuration of the experiment.
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    exp_cfg = checkpoint["hyper_parameters"]["cfg"]
    return exp_cfg, checkpoint_path
