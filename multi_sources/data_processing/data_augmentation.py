"""Defines a class and functions to augment data, meant to be applied to the data of
a single source.
"""

from typing import Any, Dict, List

import torch
import torchvision.transforms.functional as F
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform


class MultisourceDataAugmentation:
    """Class to apply data augmentation to the data of a multiple sources.
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

    def __call__(self, data, source_type):
        """Applies the augmentations to the data of each source in a batch.
        Args:
            data (dict): Dict containing the data of a single source.
            source_type (str): The type of the source.
        Returns:
            dict: The data of the source with the augmentations applied.
        """
        if source_type not in self.augmentation_functions:
            return data
        # If the source is 2D, convert the values, coords, landmask, and dist_to_center
        # to Image tv tensors so that random torchvision transform apply the same
        # transformation to all of them.
        if data["values"].ndim == 3:
            data["values"] = tv_tensors.Image(data["values"])
            data["coords"] = tv_tensors.Image(data["coords"])
            data["landmask"] = tv_tensors.Image(data["landmask"])
            data["dist_to_center"] = tv_tensors.Image(data["dist_to_center"])
        for transform in self.augmentation_functions[source_type]:
            data = transform(data)
        if data["values"].ndim == 3:
            # Convert the values, coords, landmask, and dist_to_center back to tensors
            data["values"] = data["values"].data
            data["coords"] = data["coords"].data
            data["landmask"] = data["landmask"].data
            data["dist_to_center"] = data["dist_to_center"].data
            # If the data was 2D, converting to Image added a channel dim, which
            # we don't want for the landmask and dist_to_center
            data["landmask"] = data["landmask"][0]
            data["dist_to_center"] = data["dist_to_center"][0]

        return data


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


class AddNoiseToTimeDelta:
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
            dict: The data with the modified timedelta.
        """
        # Add noise to the time delta
        dt_noise = torch.normal(mean=0.0, std=self.relative_noise_std, size=data["dt"].shape)
        # The noise cannot change the sign of the dt, so we need to
        # make sure that the noise is not larger than the dt.
        # If it is, we set the noise to the dt.
        dt_noise = torch.clamp(dt_noise, -data["dt"] + 1e-4, data["dt"] - 1e-4)

        out = {k: v for k, v in data.items() if k != "dt"}
        out["dt"] = data["dt"] + dt_noise
        return out


class RandomDynamicCrop(Transform):
    """Randomly crops the input image to a random size, with a random
    top-left corner. The crop size is a fraction of the full image size,
    and is sampled uniformly from a range defined by min_scale and max_scale.
    """

    def __init__(self, min_scale: float = 0.5, max_scale: float = 1.0):
        """
        Args:
            min_scale, max_scale: fractions of the full image size
                to bound the random crop dimensions.
        """
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        # 1️⃣ Find first image (Tensor or PIL) in flat_inputs
        img = next(x for x in flat_inputs if isinstance(x, torch.Tensor) or hasattr(x, "size"))
        # 2️⃣ Extract dimensions
        if isinstance(img, torch.Tensor):
            _, H, W = img.shape
        else:
            W, H = img.size
        # 3️⃣ Sample dynamic crop height and width
        crop_h = torch.randint(int(H * self.min_scale), H + 1, (1,)).item()
        crop_w = torch.randint(int(W * self.min_scale), W + 1, (1,)).item()
        # 4️⃣ Sample top-left corner so crop stays in-bounds
        i = torch.randint(0, H - crop_h + 1, (1,)).item()
        j = torch.randint(0, W - crop_w + 1, (1,)).item()
        return {"i": i, "j": j, "h": crop_h, "w": crop_w}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # Apply the same crop to every image-like input
        if isinstance(inpt, torch.Tensor) or hasattr(inpt, "size"):
            return F.crop(inpt, params["i"], params["j"], params["h"], params["w"])
        # Pass through all other inputs (e.g., labels) untouched
        return inpt


class CropAroundStormCenter(Transform):
    """Given the input data of a 2D source, crops the values, coords,
    landmask, and dist_to_center around the storm center. Returns the
    largest possible crop centered on the storm center. The storm
    center is defined as the pixel with the minimum value of the
    dist_to_center field.
    """

    pass
