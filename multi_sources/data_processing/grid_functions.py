"""Implements functions to manipulate gridded data."""

import logging
import traceback
import warnings
from math import ceil

import numpy as np
import torch
import xarray as xr
from haversine import haversine_vector
from numpy import nan as NA
from pyresample import SwathDefinition
from pyresample.area_config import create_area_def
from pyresample.bilinear import NumpyBilinearResampler
from pyresample.utils import check_and_wrap


class DisableLogger:
    """Context manager to disable logging temporarily."""

    def __enter__(self):
        logging.basicConfig(level=logging.ERROR, force=True)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.basicConfig(level=logging.INFO, force=True)


class ResamplingError(ValueError):
    """Exception raised when resampling operations fail."""

    pass


# Disable some warnings that pyresample and CRS are raising
# but can be safely ignored
warnings.simplefilter(action="ignore")


EARTH_RADIUS = 6371228.0  # Earth radius in meters


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


def crop_nan_border(src_image, tgt_images):
    """Computes the smallest rectangle to which a source image can be
    cropped without losing any non-NaN values; then crops the target images
    to that rectangle.

    Args:
        src_image (torch.Tensor): The source image of shape (C, H, W).
        tgt_images (list of torch.Tensor): Target images of shape (C, H, W) or (H, W).

    Returns:
        list of torch.Tensor: The cropped target images.
    """
    row_full_nan = torch.isnan(src_image).all(dim=(0, 2)).int()
    col_full_nan = torch.isnan(src_image).all(dim=(0, 1)).int()
    first_row = torch.argmax(~row_full_nan).item()
    last_row = row_full_nan.size(0) - torch.argmax(~row_full_nan.flip(dims=[0])).item() + 1
    first_col = torch.argmax(~col_full_nan).item()
    last_col = col_full_nan.size(0) - torch.argmax(~col_full_nan.flip(dims=[0])).item() + 1

    tgt_images_cropped = []
    for tgt_image in tgt_images:
        if tgt_image.ndim == 3:
            tgt_image_cropped = tgt_image[:, first_row:last_row, first_col:last_col]
        else:
            tgt_image_cropped = tgt_image[first_row:last_row, first_col:last_col]
        tgt_images_cropped.append(tgt_image_cropped)
    return tgt_images_cropped


def crop_nan_border_numpy(src_image, tgt_images):
    """Computes the smallest rectangle to which a source image can be
    cropped without losing any non-NaN values; then crops the target images
    to that rectangle. NumPy implementation.

    Args:
        src_image (numpy.ndarray): The source image of shape (H, W).
        tgt_images (list of numpy.ndarray): Target images of shape (H, W).

    Returns:
        list of numpy.ndarray: The cropped target images.
    """
    # Find rows and columns that are all NaN
    row_full_nan = np.all(np.isnan(src_image), axis=1)
    col_full_nan = np.all(np.isnan(src_image), axis=0)

    # Find first and last non-NaN rows and columns
    non_nan_rows = np.where(~row_full_nan)[0]
    non_nan_cols = np.where(~col_full_nan)[0]

    if len(non_nan_rows) == 0 or len(non_nan_cols) == 0:
        # Return originals if all NaN
        return tgt_images

    first_row = non_nan_rows[0]
    last_row = non_nan_rows[-1] + 1  # Add 1 for exclusive upper bound in slicing
    first_col = non_nan_cols[0]
    last_col = non_nan_cols[-1] + 1  # Add 1 for exclusive upper bound in slicing

    # Crop target images
    tgt_images_cropped = []
    for tgt_image in tgt_images:
        tgt_image_cropped = tgt_image[first_row:last_row, first_col:last_col]
        tgt_images_cropped.append(tgt_image_cropped)
    return tgt_images_cropped


def reverse_spatially(ds, dim_x, dim_y):
    """Reverses the dataset ds spatially, i.e. possibly reverses the spatial dimensions
    so that the latitude is decreasing from the top to the bottom of the image,
    and the longitude is increasing from left to right.

    Args:
        ds (xarray.Dataset): The dataset to reverse.
        dim_x (str): The name of the longitude dimension.
        dim_y (str): The name of the latitude dimension.
    """
    # The latitude variable in TC-PRIMEd is in degrees north, so it should be
    # decreasing from the top to the bottom of the image.
    if ds.latitude[0, 0] < ds.latitude[1, 0]:
        return ds.isel(**{dim_y: slice(None, None, -1)})
    if ds.longitude[0, 0] > ds.longitude[0, 1]:
        return ds.isel(**{dim_x: slice(None, None, -1)})
    return ds


def regrid(ds, target_resolution):
    """Regrids the dataset ds to a regular grid with a given target resolution.
    The maximum and minimum latitude and longitude values are used to define the
    target area.

    Args:
        ds (xarray.Dataset): Dataset to regrid. Must include the variables 'latitude'
            and 'longitude' and have exactly two dimensions.
        target_resolution (tuple of float): The target resolution in degrees,
            as a tuple (res_lat, res_lon).
    """
    # Get the dimensions from the latitude variable
    dims = list(ds.latitude.dims)
    if len(dims) != 2:
        raise ValueError("Dataset must have exactly two dimensions")
    dim_y, dim_x = dims

    # Rest of function remains the same but uses dim_y and dim_x
    lon, lat = check_and_wrap(ds.longitude.values, ds.latitude.values)

    swath = SwathDefinition(lons=lon, lats=lat)
    radius_of_influence = 100000  # 100 km
    sizes = ds.sizes
    size_y, size_x = sizes[dim_y], sizes[dim_x]

    # Define the Mercator projection
    central_lon = lon[size_y // 2, size_x // 2]
    proj_id = "mercator"
    proj_dict = {
        "proj": "merc",
        "lon_0": central_lon,
        "R": EARTH_RADIUS,
        "units": "m",
    }

    # Compute the corner coordinates of the target area. For the longitude, we
    # need to take into account the fact that the longitude can wrap around the
    # globe (e.g. start at 179.5 and end at -179.5).
    min_lat, max_lat = lat.min(), lat.max()
    min_lon, max_lon = lon.min(), lon.max()
    if ((min_lon < 0) and (max_lon > 0)) and (max_lon - min_lon > 180):
        # If crossing the 180° meridian
        min_lon = lon[lon > 0].min()
        max_lon = lon[lon < 0].max()
    # Round the min and max values to the nearest multiple of the target resolution
    min_lat = ceil(min_lat / target_resolution[0]) * target_resolution[0]
    max_lat = ceil(max_lat / target_resolution[0]) * target_resolution[0]
    min_lon = ceil(min_lon / target_resolution[1]) * target_resolution[1]
    max_lon = ceil(max_lon / target_resolution[1]) * target_resolution[1]
    # Define the target area object for pyresample
    with DisableLogger():  # Pyresample warns of rounded shapes, which is fine.
        target_area = create_area_def(
            proj_id,
            proj_dict,
            area_extent=[min_lon, min_lat, max_lon, max_lat],
            resolution=target_resolution,
            units="degrees",
        )

    # Retrieve the variables to regrid
    variables = [var for var in ds.variables if dim_y in ds[var].dims and dim_x in ds[var].dims]
    variables = [var for var in variables if var not in ["latitude", "longitude"]]
    ds = ds.reset_coords()[variables]

    # Create the resampler
    resampler = NumpyBilinearResampler(swath, target_area, radius_of_influence)
    # Individually resample each variable
    resampled_vars = {}
    for var in variables:
        try:
            resampled_vars[var] = resampler.resample(
                ds[var].values,
                fill_value=float("nan"),
            )
        except Exception as e:
            print("Longitude:", lon)
            print("Latitude:", lat)
            traceback.print_tb(e.__traceback__)
            traceback.print_exc()
            raise ResamplingError(f"Error resampling variable {var}") from e
    # Rebuild the datase
    result = {var: (("lat", "lon"), resampled_vars[var]) for var in variables}

    # Add the latitude and longitude variables as coordinates
    lons, lats = target_area.get_lonlats()
    coords = {
        "latitude": (["lat", "lon"], lats),
        "longitude": (["lat", "lon"], lons),
    }
    # Rebuild the dataset
    result = xr.Dataset(result, coords=coords)
    return result


def grid_distance_to_point(grid_lat, grid_lon, lat, lon):
    """Computes the distance between a point and all points of a grid.
    Args:
        grid_lat (numpy.ndarray): The latitude grid, of shape (H, W).
        grid_lon (numpy.ndarray): The longitude grid, of shape (H, W).
        lat (float): The latitude of the point.
        lon (float): The longitude of the point.
    Returns:
        numpy.ndarray: array of shape (H, W) containing the distances between
            the point and all points of the grid.
    """
    # Stack the latitude and longitude grids
    grid_latlon = np.stack([grid_lat, grid_lon], axis=-1)  # (H, W, 2)
    # Haversine expects an array of shape (N_points, 2)
    grid_latlon = grid_latlon.reshape(-1, 2)  # (H * W, 2)
    return haversine_vector([(lat, lon)], grid_latlon, comb=True).reshape(grid_lat.shape)
