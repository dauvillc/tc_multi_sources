"""Implements the AttentionRecorder class, which saves the attention weights
during decoding to a file.
Based on lucidrains' Recorder class:
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/recorder.py
"""

import numpy as np
from torch import nn
from pathlib import Path
from multi_sources.models.attention import AttentionMap


def find_last_attention_layer(model):
    attn_layers = [module for module in model.modules() if isinstance(module, AttentionMap)]
    if len(attn_layers) == 0:
        raise ValueError("No attention layers found in model.")
    return attn_layers[-1]


class AttentionRecorder(nn.Module):
    """Records the attention weights of the last attention layer during the
    forward pass, and saves them to a file.
    """

    def __init__(self, model, save_dir):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.model = model
        self.attn_layer = find_last_attention_layer(model)
        self.attn_layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # Browse the save directory. Each batch is stored
        # as <batch_idx>.npy. Find the next batch index.
        files = list(self.save_dir.glob("*.npy"))
        if len(files) == 0:
            batch_idx = 0
        else:
            batch_idx = max([int(file.stem) for file in files]) + 1
        # Write the attention weights to a new numpy file.
        attn_weights = output.cpu().detach().numpy()
        np.save(self.save_dir / f"{batch_idx}.npy", attn_weights)

    def forward(self, *args):
        return self.model(*args)


class SourceEmbeddingRecorder(nn.Module):
    # TODO
    pass
