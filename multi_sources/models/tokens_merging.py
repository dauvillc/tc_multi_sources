"""Implements layers that merge some of the tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokensMerger(nn.Module):
    """Merges adjacent tokens in a sequence to reduce the sequence length.
    Which tokens are adjacent depends on the original spatial arrangement of the
    tokens. For example, sequence that were originally images will use 2D
    windows.
    """