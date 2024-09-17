"""Implements a customized collate_fn that receives samples from a MultiSourceDataset."""

import torch


def multi_source_collate_fn(samples):
    """Collates multiple samples from a MultiSourceDataset. A sample is a dictionary D
    whose keys are source names, and values are dictionaries such that D[source_name][key]
    is either a tensor or a string.
    The samples may contain different sources, e.g. a source can be present in some samples and
    missing in others. In order to produce a batch, if a source is present in at least one sample,
    it will be included in all samples in the batch. For samples where the source is missing, the
    corresponding tensors will be filled with NaNs. An availability flag ['avail'] is added.

    Args:
        samples (list): a list of samples, where each sample is a dictionary D as described above.
    Returns:
        A dictionary D such that D[source_name] contains the information for all samples
            for the source source_name, with the entries described above.
            A entry D[source_name]['avail'] is added, which is a tensor of shape (batch_size,)
            containing 1 where the source is available and -1 where it is missing.
    """
    # Browse the samples. For each source in the sample, browse the keys and store the type
    # of the value. If it is a tensor, store the shape as well. If it is a string, store its
    # value.
    source_key_types = {}
    source_key_shapes = {}
    source_key_str_values = {}
    for sample in samples:
        for source_name, source_dict in sample.items():
            if source_name not in source_key_types:
                source_key_types[source_name] = {}
                source_key_shapes[source_name] = {}
                source_key_str_values[source_name] = {}
            for key, value in source_dict.items():
                if key not in source_key_types[source_name]:
                    source_key_types[source_name][key] = type(value)
                    if isinstance(value, torch.Tensor):
                        source_key_shapes[source_name][key] = value.shape
                    elif isinstance(value, str):
                        source_key_str_values[source_name][key] = value

    # Create the batch dictionary
    batch = {}
    # Browse the sources that have been found. For each source, browse the keys.
    # For each key, browse the samples. For samples where the source is missing,
    # if the key is a tensor, fill it with NaNs. If the key is a string, fill it
    # with the stored value.
    # Then concatenate the tensors along the first dimension to form the batch
    # for that source and key.
    for source_name, key_types in source_key_types.items():
        batch[source_name] = {}
        for key, value_type in key_types.items():
            batch_list = []
            if value_type == torch.Tensor:
                for sample in samples:
                    if source_name in sample and key in sample[source_name]:
                        batch_list.append(sample[source_name][key])
                    else:
                        val = torch.full(source_key_shapes[source_name][key], float("nan"))
                        batch_list.append(val)
                batch[source_name][key] = torch.stack(batch_list)
            elif value_type == str:
                for sample in samples:
                    if source_name in sample and key in sample[source_name]:
                        batch_list.append(sample[source_name][key])
                    else:
                        batch_list.append(source_key_str_values[source_name][key])
                batch[source_name][key] = batch_list
            else:
                raise ValueError(f"Unsupported value type {value_type}")
        # Add the availability flag
        batch[source_name]["avail"] = torch.tensor(
            [1 if source_name in sample else -1 for sample in samples]
        )
    return batch