"""Defines a class and functions to augment data, meant to be applied to the data of
a single source.
"""

import torch
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors


class MultisourceDataAugmentation:
    """Class to apply data augmentation to the data of a multiple sources. The augmentations
    are applied to each source separately, and which augmentations are applied to each source
    depends on the dimensionality of the source.
    The data of a single source is a dict with the keys:
    - "source_type" is a string containing the type of the source.
    - "avail" is a scalar tensor of shape (1,) containing 1 if the element is available
        and -1 otherwise.
    - "dt" is a scalar tensor of shape (1,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "coords" is a tensor of shape (2, H, W) containing the latitude and longitude at each pixel.
    - "landmask" is a tensor of shape (H, W) containing the land mask.
    - "dist_to_center" is a tensor of shape (H, W) containing the distance
        to the center of the storm.
    - "values" is a tensor of shape (n_variables, H, W) containing the variables for the source.

    If a transformation inserts missing values into the data, they should be encoded with NaNs, and
    will be dealt with by the lightning module.
    """

    def __init__(self, augmentation_functions):
        """
        Args:
            augmentation_functions (dict of str: Sequence[Callable]): Dict {source_type: [f1, f2, ...]}
                where source_type is the type of the source, and f1, f2, ... are functions
                that apply the augmentations to the data of the source.
                Each function should have the signature `f(data) -> data`, where data is a dict
                with the keys described above.
                wrap_tv_image_transform can be used to wrap torchvision transforms into functions
                of the required signature.
        """
        self.augmentation_functions = augmentation_functions

    def __call__(self, batch):
        """Applies the augmentations to the data of each source in a batch.
        Args:
            batch (dict of str: dict): Dictionary of sources, such that
                batch[source_name] contains the data of the source.
        Returns:
            dict of str: dict: Dictionary of sources, such that
                batch[source_name] contains the augmented data of the source.
        """
        for source_name, data in batch.items():
            s_type = data["source_type"]
            if s_type not in self.augmentation_functions:
                continue
            # If the source is 2D, convert the values, coords, landmask, and dist_to_center
            # to Image tv tensors so that random torchvision transform apply the same
            # transformation to all of them.
            if data["values"].ndim == 3:
                data['values'] = tv_tensors.Image(data['values'])
                data['coords'] = tv_tensors.Image(data['coords'])
                data['landmask'] = tv_tensors.Image(data['landmask'])
                data['dist_to_center'] = tv_tensors.Image(data['dist_to_center'])
            for transform in self.augmentation_functions[s_type]:
                data = transform(data)
            if data["values"].ndim == 3:
                # Convert the values, coords, landmask, and dist_to_center back to tensors
                data['values'] = data['values'].data
                data['coords'] = data['coords'].data
                data['landmask'] = data['landmask'].data
                data['dist_to_center'] = data['dist_to_center'].data
                # If the data was 2D, converting to Image added a channel dim, which
                # we don't want for the landmask and dist_to_center
                data['landmask'] = data['landmask'][0]
                data['dist_to_center'] = data['dist_to_center'][0]
            batch[source_name] = data

        return batch


def wrap_tv_image_transform(transform):
    """Decorator to wrap a torchvision image transform into a function that
    applies the transform to the data of a source.
    Args:
        transform (torchvision.transforms.Transform): The transform to apply.
    Returns:
        function: The function that applies the transform to the data.
    """

    def wrapped(data):
        # Retrieve the values, coords, landmask, and dist_to_center
        values = data["values"]
        coords = data["coords"]
        landmask = data["landmask"]
        dist_to_center = data["dist_to_center"]
        # Apply the transform
        values, coords, landmask, dist_to_center = transform(
            values, coords, landmask, dist_to_center
        )
        output = {}
        output["values"] = values
        output["coords"] = coords
        output["landmask"] = landmask
        output["dist_to_center"] = dist_to_center
        # Copy the other fields of the data
        for field in data:
            if field not in output:
                output[field] = data[field]
        return output

    return wrapped


class Add_noise_to_timedelta:
    """Adds noise to the timedelta of the source, to prevent
    the model from learning the sample based on the dt.
    """

    def __init__(self, relative_noise_std):
        """
        Args:
            relative_noise_std (float): The standard deviation of the noise,
                relative to the maximum timedelta which is set to 1. For example,
                if the true dt_max is 24h, then a relative std of 0.5 corresponds
                to 12h.
        """
        self.relative_noise_std = relative_noise_std

    def __call__(self, data):
        """
        Args:
            data (dict): The data of the source.
        Returns:
            dict: The data with the modified timedelta, which is also modified
                in-place.
        """
        # Add noise to the time delta
        dt_noise = torch.normal(
            mean=0.0, std=self.relative_noise_std, size=data["dt"].shape
        ).item()
        data["dt"] += dt_noise

        return data
