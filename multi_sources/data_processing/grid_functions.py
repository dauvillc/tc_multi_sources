"""Implements functions to manipulate gridded data."""
from xarray.core.dtypes import NA


def pad_dataset(ds, max_size):
    """Pads the dataset ds to the size max_size, by adding missing values
    at the bottom and right of the dataset.
    If the dataset is already larger than max_size, crops it to max_size.

    Args:
        ds (xarray.Dataset): The dataset to pad.
        max_size (tuple): The maximum size of the dataset, as a tuple (scan, pixel).
    """
    # Retrieve the size of the dataset
    sizes = ds.sizes
    size = (sizes["scan"], sizes["pixel"])
    # Compute the padding values
    pad_scan = max_size[0] - size[0]
    pad_pixel = max_size[1] - size[1]
    # Set all -9999.9 values to NaN
    ds = ds.where(ds != -9999.9)
    # Along the scan dimension, pad the dataset with NaN values if pad_scan > 0
    # or crop it if pad_scan < 0
    if pad_scan > 0:
        ds = ds.pad(scan=(0, pad_scan), mode="constant", constant_values=NA)
    elif pad_scan < 0:
        ds = ds.isel(scan=slice(None, max_size[0]))
    # Same for the pixel dimension
    if pad_pixel > 0:
        ds = ds.pad(pixel=(0, pad_pixel), mode="constant", constant_values=NA)
    elif pad_pixel < 0:
        ds = ds.isel(pixel=slice(None, max_size[1]))
    return ds


def reverse_spatially(ds):
    """Reverses the dataset ds spatially, i.e. possibly reverses the scan and pixel
    dimensions so that the latitude is decreasing from the top to the bottom of the image,
    and the longitude is increasing from left to right.

    Args:
        ds (xarray.Dataset): The dataset to reverse.
    """
    # The latitude variable in TC-PRIMEd is in degrees north, so it should be
    # decreasing from the top to the bottom of the image.
    if ds.latitude[0, 0] < ds.latitude[1, 0]:
        return ds.isel(scan=slice(None, None, -1))
    if ds.longitude[0, 0] > ds.longitude[0, 1]:
        return ds.isel(pixel=slice(None, None, -1))
    return ds
