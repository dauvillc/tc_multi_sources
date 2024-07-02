"""Implements functions to manipulate gridded data."""

import numpy as np
import xarray as xr
import warnings
from math import ceil
from xarray.core.dtypes import NA
from pyresample.bilinear.xarr import XArrayBilinearResampler
from pyresample import SwathDefinition
from pyresample.area_config import create_area_def
from pyresample.utils import check_and_wrap


# Disable some warnings that pyresample and CRS are raising
# but can be safely ignored
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


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


def regrid(ds, target_resolution_km, target_area):
    """Regrids the dataset ds to a regular grid with a given target resolution.
    The resulting area is centered on the original area's central coordinates.

    Args:
        ds (xarray.Dataset): Dataset to regrid. Must include the variables 'latitude'
            and 'longitude', along the dimensions 'scan' and 'pixel'.
            All variables in the dataset will be regridded.
        target_resolution_km (tuple of float): The target resolution in kilometers,
            as a tuple (res_lon, res_lat).
        target_area (tuple of float): Tuple (d_lon, d_lat) representing the size of the
            target area in degrees.
    """
    # Compute the radius of influence for the regridding (in meters)
    radius_of_influence = (
        np.sqrt(target_resolution_km[0] ** 2 + target_resolution_km[1] ** 2) * 20000
    )
    # Reformat the coordinates to suit pyresample
    lon, lat = check_and_wrap(ds.longitude.values, ds.latitude.values)
    # Define the swath
    swath = SwathDefinition(lons=lon, lats=lat)
    # Retrieve the size of the dataset
    sizes = ds.sizes
    size_scan, size_pixel = sizes["scan"], sizes["pixel"]
    # Compute the target resolution in degrees
    target_resolution = (target_resolution_km[0] / 111.32, target_resolution_km[1] / 111.32)
    # Compute the target area's central coordinates
    central_lat = ds.latitude[size_scan // 2, size_pixel // 2].values
    central_lon = ds.longitude[size_scan // 2, size_pixel // 2].values
    # Define the Mercator projection
    proj_id = "mercator"
    proj_dict = {
        "proj": "merc",
        "lon_0": 0,
        "R": EARTH_RADIUS,
        "units": "m",
    }
    # Compute the target size in pixels
    target_size = (
        ceil(target_area[0] / target_resolution[0]),
        ceil(target_area[1] / target_resolution[1]),
    )
    # Define the target area
    target_area = create_area_def(
        proj_id,
        proj_dict,
        center=(central_lon, central_lat),
        resolution=target_resolution,
        shape=target_size,
        units="degrees",
    )
    # Retrieve the variables to regrid
    variables = [var for var in ds.variables if "scan" in ds[var].dims and "pixel" in ds[var].dims]
    variables = [var for var in variables if var not in ["latitude", "longitude"]]
    ds = ds[variables].reset_coords().rename({"scan": "y", "pixel": "x"})
    # Create the resampler
    resampler = XArrayBilinearResampler(
        swath, target_area, radius_of_influence=radius_of_influence
    )
    # Resample each variable
    result = {var: resampler.resample(ds[var]) for var in variables}
    result = xr.Dataset(result)
    # Rename the 'x' and 'y' dimensions to 'latitude' and 'longitude'
    result = result.rename({"x": "longitude", "y": "latitude"})
    # Add the latitude and longitude variables
    lons, lats = target_area.get_lonlats()
    result["latitude"] = xr.DataArray(lats, dims=("latitude", "longitude"))
    result["longitude"] = xr.DataArray(lons, dims=("latitude", "longitude"))
    return result
