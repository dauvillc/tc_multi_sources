#!/usr/bin/env python3

"""
Prepares the ERA5 data for the TC Primed dataset.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from global_land_mask import globe
from netCDF4 import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from xarray.backends import NetCDF4DataStore

from multi_sources.data_processing.grid_functions import (
    grid_distance_to_point,
)

# Local imports
from preproc.utils import list_tc_primed_sources

DATA_VARS = ["pressure_msl", "u_wind_10m", "v_wind_10m", "sst"]


def initialize_metadata(dest_dir):
    """Initializes the metadata for the ERA5 data.
    Args:
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """

    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": "tc_primed_era5",
        "source_type": "era5",
        "dim": 2,
        "data_vars": DATA_VARS,
        "storm_metadata_vars": storm_vars,
        "is_on_regular_grid": True,
        "charac_vars": {},
    }
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)
    return True


def process_era5_file(file, dest_dir, patch_size, check_exist=False):
    """Processes a single ERA5 file.
    Args:
        file (str): Path to the input file.
        dest_dir (Path): Destination directory.
        patch_size (float): Size of the ERA5 patches in pixels.
        check_exist (bool): Flag to check if the file already exists.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
    """
    with Dataset(file) as raw_sample:
        # Access data group
        all_times_ds = raw_sample["rectilinear"]
        # Open the dataset using xarray and select the brightness temperature variable
        all_times_ds = xr.open_dataset(NetCDF4DataStore(all_times_ds), decode_times=False)
        # The dataset has a "time" dimension, and we'll need to process each time step.
        times = all_times_ds["time"].values

        # Assemble the sample's metadata
        storm_meta = raw_sample["storm_metadata"]
        season = storm_meta["season"][0].item()
        basin = storm_meta["basin"][0]
        storm_number = storm_meta["cyclone_number"][-1].item()
        sid = f"{season}{basin}{storm_number}"  # Unique storm identifier

        # Process each time step
        samples_metadata = []
        for time_idx, time in enumerate(times):
            time = pd.to_datetime(time, unit="s", origin="unix")
            storm_lat = storm_meta["storm_latitude"][time_idx].item()
            storm_lon = storm_meta["storm_longitude"][time_idx].item()
            storm_lon = (storm_lon + 180) % 360 - 180
            storm_intensity = storm_meta["intensity"][time_idx].item()

            sample_metadata = {
                "source_name": "tc_primed_era5",
                "source_type": "era5",
                "sid": sid,
                "time": time,
                "season": season,
                "storm_latitude": storm_lat,
                "storm_longitude": storm_lon,
                "intensity": storm_intensity,
                "storm_number": storm_number,
                "basin": basin,
                "dim": 2,  # Spatial dimensionality
            }
            dest_file = dest_dir / f"{sid}_{time.strftime('%Y%m%dT%H%M%S')}.nc"
            sample_metadata["data_path"] = dest_file

            # Check if the file already exists. If it does, skip processing.
            if check_exist and dest_file.exists():
                return sample_metadata

            # Select the dataset at the current time step
            ds = all_times_ds.isel(time=time_idx)
            ds = ds[DATA_VARS + ["latitude", "longitude"]]
            # Rename the "ny" and "nx" dimensions to "lat" and "lon"
            ds = ds.rename({"ny": "lat", "nx": "lon"})

            # Padding:
            # - Create a new lat/lon grid with the same center but that cover patch_size degrees
            try:
                res, old_lat, old_lon = 0.25, ds["latitude"].values, ds["longitude"].values
                old_lat, old_lon = old_lat.round(2), old_lon.round(2)
                center_idx = old_lat.size // 2
                lat_center, lon_center = (
                    old_lat[center_idx],
                    old_lon[center_idx],
                )
                new_lat = np.arange(
                    max(lat_center - (patch_size // 2) * res, -90),
                    min(lat_center + (patch_size // 2 + 1) * res, 90),
                    res,
                ).round(2)
                new_lon = np.arange(
                    lon_center - (patch_size // 2) * res,
                    lon_center + (patch_size // 2 + 1) * res,
                    res,
                ).round(2)
                # Standardize longitude values to [-180, 180]
                new_lon = (new_lon + 180) % 360 - 180
                # We'll pad the dataset with NaNs and then set the new latitude and longitude values
                # For the latitude, we must recompute the padding because the latitudes may have
                # been clipped at the poles.
                patch_lat_before = int((old_lat[0] - new_lat[0]) / res)
                patch_lat_after = int((new_lat[-1] - old_lat[-1]) / res)
                ds = ds.pad(
                    lat=(patch_lat_before, patch_lat_after),
                    lon=(new_lon.size - old_lon.size) // 2,
                    method="constant",
                    constant_values=np.nan,  # by default but for clarity
                )
                ds = ds.assign_coords(latitude=("lat", new_lat), longitude=("lon", new_lon))
            except Exception as e:
                print("Old latitudes:", old_lat, old_lat.shape)
                print("Old longitudes:", old_lon, old_lon.shape)
                print("New latitudes:", new_lat, new_lat.shape)
                print("New longitudes:", new_lon, new_lon.shape)
                print(ds)
                print(ds["pressure_msl"])
                print("New size:", new_lat.size, new_lon.size)
                raise e

            # Reverse the latitude dimension to have higher latitudes at the top
            # after the next step.
            ds = ds.isel(lat=slice(None, None, -1))
            # At this point the lat and lon variables are 1D arrays, but
            # we need to make them 2D arrays which give the lat/lon at each
            # grid point.
            lat, lon = ds["latitude"].values, ds["longitude"].values
            ds["latitude"] = ds["latitude"].expand_dims(dim={"lon": lon.shape[0]}, axis=1)
            ds["longitude"] = ds["longitude"].expand_dims(dim={"lat": lat.shape[0]}, axis=0)

            # Compute the land-sea mask
            land_mask = globe.is_land(
                ds["latitude"].values,
                ds["longitude"].values,
            )
            # Compute the distance between each grid point and the center of the storm.
            dist_to_center = grid_distance_to_point(
                ds["latitude"].values,
                ds["longitude"].values,
                storm_lat,
                storm_lon,
            )
            # Add the land mask and distance to center as new variables.
            ds["land_mask"] = (("lat", "lon"), land_mask)
            ds["dist_to_center"] = (("lat", "lon"), dist_to_center)

            # Save processed data in the netCDF format
            ds = ds[DATA_VARS + ["latitude", "longitude", "land_mask", "dist_to_center"]]
            ds = ds.drop_encoding()  # Drop the "unlimited_dims" encoding on the "time" dim.
            ds.to_netcdf(dest_file)

            samples_metadata.append(sample_metadata)
        return samples_metadata


# Helper function to process a chunk of files in parallel.
def process_chunk(file_list, source_dest_dir, patch_size, check_exist, verbose=False):
    local_metadata = []
    discarded = 0

    iterator = tqdm(file_list, desc="Processing chunk") if verbose else file_list
    for file in iterator:
        meta = process_era5_file(file, source_dest_dir, patch_size, check_exist)
        if meta is None:
            discarded += 1
        else:
            local_metadata += meta
    return local_metadata, discarded


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(cfg):
    """Main function to process IR data."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    patch_size = cfg.get("era5_patch_size", 201)  # Default 50 degrees at 0.25° resolution

    # Setup paths
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"
    check_exist = cfg.get("check_exist", False)

    # Get environmental files only
    sources, source_files, source_groups = list_tc_primed_sources(
        tc_primed_path,
        source_type="environmental",
    )
    era5_files = source_files.get("era5", [])

    source_dest_dir = dest_path / "tc_primed_era5"
    # Create destination directory
    source_dest_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metadata
    if not initialize_metadata(source_dest_dir):
        raise RuntimeError("Failed to initialize metadata.")

    # Process all files, and keep count of discarded files and save the metadata of
    # each sample.
    discarded, samples_metadata = 0, []
    num_workers = cfg.get("num_workers", 1)

    if num_workers > 1:
        chunks = np.array_split(era5_files, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(
                        process_chunk,
                        list(chunk),
                        source_dest_dir,
                        patch_size,
                        check_exist,
                        verbose=(i == 0),
                    )
                )
            for future in as_completed(futures):
                meta_chunk, discarded_chunk = future.result()
                samples_metadata.extend(meta_chunk)
                discarded += discarded_chunk
    else:
        for file in tqdm(era5_files, desc="Processing files"):
            file_times_metadata = process_era5_file(file, source_dest_dir, patch_size, check_exist)
            if file_times_metadata is None:
                discarded += 1
            else:
                samples_metadata += file_times_metadata

    # If all files were discarded, remove all files inside the directory
    # and remove the directory itself.
    if discarded == len(era5_files):
        for file in source_dest_dir.iterdir():
            file.unlink()
        source_dest_dir.rmdir()
        print("Found no valid samples, removing directory.")
    else:
        if discarded > 0:
            percent = discarded / len(era5_files) * 100
            print(f"Discarded {discarded} samples ({percent:.2f}%)")
        # Concatenate the samples metadata into a DataFrame and save it
        samples_metadata = pd.DataFrame(samples_metadata)
        samples_metadata_path = source_dest_dir / "samples_metadata.csv"
        samples_metadata.to_csv(samples_metadata_path, index=False)


if __name__ == "__main__":
    main()
