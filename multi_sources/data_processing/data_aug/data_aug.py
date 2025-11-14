"""Defines a class and functions to augment data, meant to be applied to the data of
a single source.
"""


class MultisourceDataAugmentation:
    """Class to apply data augmentation to the data of a multiple sources.
    Expects samples from the dataset as dicts {(source_name, index): data} where
    source_name is the name of the source and index is an integer representing
    the observation index (0 = most recent, 1 = second most recent, etc.).
    Each data dict contains the following key-value pairs:
    - "avail" is a scalar tensor of shape (1,) containing 1 if the element is available
        and -1 otherwise.
    - "dt" is a scalar tensor of shape (1,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "characs" is a tensor of shape (n_charac_vars,) containing the source characteristics.
        Each data variable within a source has its own charac variables, which are all
        concatenated into a single tensor.
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
            augmentation_functions (list of Callable): List [f1, f2, ...]
                where f1, f2, ... are functions that apply the augmentations to the full
                data (all sources).
                Each function should have the signature `f(data) -> data`, where data is a dict
                with the keys described above.
        """
        self.augmentation_functions = augmentation_functions

    def __call__(self, data):
        """Applies the augmentations to the data of each source in a batch.
        Args:
            data (dict): Dict containing the data of all sources.
        Returns:
            dict: The data with the augmentations applied.
        """
        for func in self.augmentation_functions:
            data = func(data)
        return data
