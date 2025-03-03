"""Implements a customized collate_fn that receives samples from a MultiSourceDataset."""

import torch
from torch.nn.functional import pad
from collections import defaultdict


def maximum_of_shapes(shape_A, shape_B):
    """Returns the maximum shape dim per dim between shape_A and shape_B."""
    if not shape_A:
        return shape_B
    if not shape_B:
        return shape_A
    return tuple(max(a, b) for a, b in zip(shape_A, shape_B))


def multi_source_collate_fn(samples):
    """Collates multiple samples from a MultiSourceDataset. A sample is a dictionary D
    whose keys are source names, and values are dictionaries such that D[S][k] is the tensor
    for the key k of source S.
    The samples may contain different sources, e.g. a source can be present in some samples and
    missing in others. In order to produce a batch, if a source is present in at least one sample,
    it will be included in all samples in the batch. For samples where the source is missing, the
    corresponding tensors will be filled with NaNs. An availability flag ['avail'] is added,
    containing 1 where the source is available and -1 where it is missing. Samples of the same
    key in the same source are padded with NaNs to match the maximum shape across all samples,
    so that they can be stacked in a batch.

    Args:
        samples (list): a list of samples, where each sample is a dictionary D as described above.
    Returns:
        A dictionary D such that D[source_name] contains the information for all samples
            for the source source_name, with the entries described above.
    """
    # Browse the samples. For each source in the sample, find the keys that can be found
    # in that source.
    source_keys = defaultdict(set)
    for sample in samples:
        for source_name, source_dict in sample.items():
            source_keys[source_name].update(source_dict.keys())

    # Create the batch dictionary
    batch = {}
    # Pseudo-code:
    # For each source found:
    #   For each key found in the source:
    #       - Compute the maximum shape across all samples
    #       - Create an empty list L to store the tensors
    #       - For each sample:
    #         If the sample contains that key for that source, pad it with NaNs
    #           to match the maximum spatial shape, and store it in L.
    #         Else, create a tensor of NaNs with the maximum spatial shape
    #          and store it in L.
    #       - Stack the tensors.
    for source_name, keys in source_keys.items():
        batch[source_name] = {}
        for key in keys:
            # - Compute the maximum shape across all samples
            max_shape = None
            for sample in samples:
                if source_name in sample and key in sample[source_name]:
                    max_shape = maximum_of_shapes(max_shape, sample[source_name][key].shape)
            #  - Assemble the batch
            L = []
            for sample in samples:
                if source_name in sample and key in sample[source_name]:
                    # The key is present for that source in that sample
                    val = sample[source_name][key]
                    # Pad the tensor with NaNs. Torch's F.pad() expects
                    # (last_dim_before, last_dim_after, ..., first_dim_before, first_dim_after)
                    pad_shape = []
                    for ms, vs in zip(max_shape[::-1], val.shape[::-1]):
                        pad_shape.append(0)
                        pad_shape.append(ms - vs)
                    val = pad(val, pad_shape, value=float("nan"))
                    L.append(val)
                else:
                    # The key is missing for that source in that sample
                    L.append(torch.full(max_shape, float("nan")))
            batch[source_name][key] = torch.stack(L)
        # Add the availability flag
        batch[source_name]["avail"] = torch.tensor(
            [1 if source_name in sample else -1 for sample in samples]
        )
    return batch
