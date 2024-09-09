"""Implements functions to manipulate gridded data."""

import xarray as xr
import warnings
import numpy as np
from haversine import haversine_vector
from numpy import nan as NA
from math import ceil
from pyresample.kd_tree import resample_nearest
from pyresample import SwathDefinition
from pyresample.area_config import create_area_def
from pyresample.utils import check_and_wrap


# Disable some warnings that pyresample and CRS are raising
# but can be safely ignored
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


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
    # Reformat the coordinates to suit pyresample
    lon, lat = check_and_wrap(ds.longitude.values, ds.latitude.values)
    # Define the swath
    swath = SwathDefinition(lons=lon, lats=lat)
    radius_of_influence = 100000  # 100 km
    # Retrieve the size of the dataset
    sizes = ds.sizes
    size_scan, size_pixel = sizes["scan"], sizes["pixel"]
    # Compute the target resolution in degrees
    target_resolution = (target_resolution_km[0] / 111.32, target_resolution_km[1] / 111.32)
    # Compute the target area's central coordinates
    central_lat = lat[size_scan // 2, size_pixel // 2]
    central_lon = lon[size_scan // 2, size_pixel // 2]
    # Define the Mercator projection
    proj_id = "mercator"
    proj_dict = {
        "proj": "merc",
        "lon_0": central_lon,
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
    ds = ds.reset_coords()[variables]
    # Resample all variables at once, by stacking them along the last dimension
    stacked = ds.to_dataarray().transpose("scan", "pixel", "variable").values
    stacked = resample_nearest(
        swath,
        stacked,
        target_area,
        radius_of_influence=radius_of_influence,
        fill_value=float("nan"),
    )

    result = {var: (("lat", "lon"), stacked[..., i]) for i, var in enumerate(variables)}
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
