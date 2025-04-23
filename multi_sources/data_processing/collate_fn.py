"""Implements a customized collate_fn that receives samples from a MultiSourceDataset."""

from collections import defaultdict

import torch
from torch.nn.functional import pad


def maximum_of_shapes(shape_A, shape_B):
    """Returns the maximum shape dim per dim between shape_A and shape_B."""
    if not shape_A:
        return shape_B
    if not shape_B:
        return shape_A
    return tuple(max(a, b) for a, b in zip(shape_A, shape_B))


def multi_source_collate_fn(samples):
    """Collates multiple samples from a MultiSourceDataset. A sample is a dictionary D
    whose keys are (source_name, index) tuples, and values are dictionaries such that
    D[(S, I)][k] is the tensor for the key k of source S with observation index I.

    The samples may contain different sources or multiple observations of the same source
    with different indices. If a source-index pair is present in at least one sample but missing
    in others, it will be included in all samples in the batch with NaN values and an
    availability flag ['avail'] = -1. Samples of the same key in the same source-index
    pair are padded with NaNs to match the maximum shape across all samples,
    so that they can be stacked in a batch.

    Args:
        samples (list): a list of samples, where each sample is a dictionary D as described above.
    Returns:
        A dictionary D such that D[(source_name, index)] contains the information for all samples
            for the source source_name with observation index index, with the entries described above.
    """
    # Browse the samples. For each source-index pair in the sample, find the keys that
    # can be found in that source.
    source_keys = defaultdict(set)
    for sample in samples:
        for source_index_pair, source_dict in sample.items():
            source_keys[source_index_pair].update(source_dict.keys())

    # Create the batch dictionary
    batch = {}
    # For each source-index pair found:
    #   For each key found in the source:
    #       - Compute the maximum shape across all samples
    #       - Create an empty list L to store the tensors
    #       - For each sample:
    #         If the sample contains that key for that source-index pair, pad it with NaNs
    #           to match the maximum spatial shape, and store it in L.
    #         Else, create a tensor of NaNs with the maximum spatial shape
    #          and store it in L.
    #       - Stack the tensors.
    for source_index_pair, keys in source_keys.items():
        batch[source_index_pair] = {}
        for key in keys:
            # - Compute the maximum shape across all samples
            max_shape = None
            for sample in samples:
                if source_index_pair in sample and key in sample[source_index_pair]:
                    max_shape = maximum_of_shapes(max_shape, sample[source_index_pair][key].shape)
            #  - Assemble the batch
            L = []
            for sample in samples:
                if source_index_pair in sample and key in sample[source_index_pair]:
                    # The key is present for that source-index pair in that sample
                    val = sample[source_index_pair][key]
                    # Pad the tensor with NaNs. Torch's F.pad() expects
                    # (last_dim_before, last_dim_after, ..., first_dim_before, first_dim_after)
                    pad_shape = []
                    for ms, vs in zip(max_shape[::-1], val.shape[::-1]):
                        pad_shape.append(0)
                        pad_shape.append(ms - vs)
                    val = pad(val, pad_shape, value=float("nan"))
                    L.append(val)
                else:
                    # The key is missing for that source-index pair in that sample
                    L.append(torch.full(max_shape, float("nan")))
            batch[source_index_pair][key] = torch.stack(L)
        # Add the availability flag
        batch[source_index_pair]["avail"] = torch.tensor(
            [1 if source_index_pair in sample else -1 for sample in samples]
        )
    return batch
