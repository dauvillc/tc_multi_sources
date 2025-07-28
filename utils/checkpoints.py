"""Implements small utility functions for loading checkpoints."""

from pathlib import Path

import torch


def load_experiment_cfg_from_checkpoint(checkpoints_dir, run_id, best_or_latest="latest"):
    """Loads a Lightning checkpoint and extracts from it the
    experiment configuration (which must have been saved as the "cfg"
    hyperparameter of the LightningModule).

    Args:
        checkpoints_dir (str): The directory where the checkpoints are stored.
        run_id (str): The ID of the run.
        best_or_latest (str): Whether to load the best or latest checkpoint.
    Returns:
        exp_cfg (dict): The experiment configuration.
        checkpoint_path (str): The path to the checkpoint.
    """
    checkpoints_dir = Path(checkpoints_dir) / run_id
    checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
    # The checkpoints are stored as:
    # - <run_id>-<epoch>-<step>.ckpt for the timed checkpoints
    # - <run_id>-<epoch>-best.ckpt for the best checkpoint
    if best_or_latest == "latest":
        # Filter out the best checkpoints if we want the latest one.
        checkpoint_files = [f for f in checkpoint_files if "best" not in f.stem]
    # If no "latest" checkpoints are found, we load the best one.
    if best_or_latest == "best" or checkpoint_files == []:
        checkpoint_files = [f for f in checkpoint_files if "best" in f.stem]
    # Load the latest checkpoint out of the filtered list.
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
    checkpoint_path = checkpoint_files[-1]
    print("Loading checkpoint:", checkpoint_path.stem, " from run ", run_id)
    # The checkpoint includes the whole configuration dict of the experiment, in
    # checkpoint["hyper_parameters"]['cfg']. We'll use this to reproduce
    # the exact configuration of the experiment.
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    exp_cfg = checkpoint["hyper_parameters"]["cfg"]
    return exp_cfg, checkpoint_path


def load_weights_intersection(former_dict, current_dict):
    """Loads the weights of the intersection of the keys in the
    former and current dictionaries.

    Args:
        former_dict (dict): The former dictionary.
        current_dict (dict): The current dictionary.
    Returns:
        new_dict (dict): The new dictionary with the weights of the intersection.
    """
    new_dict = {}
    for k, v in current_dict.items():
        if k in former_dict:
            new_dict[k] = former_dict[k]
        else:
            new_dict[k] = v
    return new_dict
