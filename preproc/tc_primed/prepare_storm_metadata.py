#!/usr/bin/env python3

"""
Prepares the storm metadata for the TC Primed dataset.
"""

import hydra
import yaml
import pandas as pd
import xarray as xr
import json
import warnings
from tqdm import tqdm
from netCDF4 import Dataset
from xarray.backends import NetCDF4DataStore
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from global_land_mask import globe
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
from preproc.utils import list_tc_primed_sources

DATA_VARS = ["intensity", "central_min_pressure", "storm_speed", "storm_heading"]


def initialize_metadata(dest_dir):
    """Initializes the metadata for the storm data.
    Args:
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """

    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": "tc_primed_storm_metadata",
        "source_type": "storm_metadata",
        "dim": 0,
        "data_vars": DATA_VARS,
        "storm_metadata_vars": storm_vars,
        "is_on_regular_grid": True,
        "charac_vars": {},
    }
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)
    return True


def process_storm_data_file(file, dest_dir, check_exist=False):
    """Processes a single storm metadata file.
    Args:
        file (str): Path to the input file.
        dest_dir (Path): Destination directory.
        check_exist (bool): Flag to check if the file already exists.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
    """
    with Dataset(file) as raw_sample:
        # Access data group
        all_times_ds = raw_sample["storm_metadata"]
        # Open the dataset using xarray and select the brightness temperature variable
        all_times_ds = xr.open_dataset(NetCDF4DataStore(all_times_ds), decode_times=False)
        # The dataset has a "time" dimension, and we'll need to process each time step.
        times = all_times_ds["time"].values

        # Assemble the sample's metadata
        season = all_times_ds["season"].item()
        basin = all_times_ds["basin"].item()
        storm_number = all_times_ds["cyclone_number"].item()
        sid = f"{season}{basin}{storm_number}"  # Unique storm identifier

        # Process each time step
        samples_metadata = []
        for time_idx, time in enumerate(times):
            time = pd.to_datetime(time, unit="s", origin="unix")
            # Select the dataset at the current time step
            ds = all_times_ds.isel(time=time_idx)
            storm_lat = ds["storm_latitude"].item()
            storm_lon = ds["storm_longitude"].item()
            storm_lon = (storm_lon + 180) % 360 - 180
            storm_intensity = ds["intensity"].item()
            ds = ds[DATA_VARS]
            ds["latitude"] = storm_lat
            ds["longitude"] = storm_lon

            sample_metadata = {
                "source_name": "tc_primed_storm_metadata",
                "source_type": "storm_metadata",
                "sid": sid,
                "time": time,
                "season": season,
                "basin": basin,
                # Lat and lon as storm metadata
                "storm_latitude": storm_lat,
                "storm_longitude": storm_lon,
                "intensity": storm_intensity,
                "dim": 0,  # Spatial dimensionality
            }
            dest_file = dest_dir / f"{sid}_{time.strftime('%Y%m%dT%H%M%S')}.nc"
            sample_metadata["data_path"] = dest_file

            # Check if the file already exists. If it does, skip processing.
            if check_exist and dest_file.exists():
                samples_metadata.append(sample_metadata)
                continue

            # Compute the land-sea mask
            land_mask = globe.is_land(
                ds["latitude"].values,
                ds["longitude"].values,
            )
            # Compute the distance between each grid point and the center of the storm.
            dist_to_center = np.zeros(ds["latitude"].shape)
            # Add the land mask and distance to center as new variables.
            ds["land_mask"] = land_mask
            ds["dist_to_center"] = dist_to_center

            # Select only the relevant variables
            ds = ds[DATA_VARS + ["latitude", "longitude", "land_mask", "dist_to_center"]]
            
            # If any entry is NaN, discard the sample.
            if ds.isnull().any():
                print(f"Discarding sample {sid} at time {time} due to NaN values.")
                continue

            # Save the dataset to a NetCDF file
            ds.to_netcdf(dest_file)

            samples_metadata.append(sample_metadata)
        return samples_metadata


# Helper function to process a chunk of files in parallel.
def process_chunk(file_list, source_dest_dir, check_exist, verbose=False):
    local_metadata = []
    discarded = 0

    iterator = tqdm(file_list, desc="Processing chunk") if verbose else file_list
    for file in iterator:
        meta = process_storm_data_file(file, source_dest_dir, check_exist)
        if meta is None:
            discarded += 1
        else:
            local_metadata += meta
    return local_metadata, discarded


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(cfg):
    """Main function to process IR data."""
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup paths
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"
    # Resolution of the target grid, in degrees
    regridding_res = cfg["regridding_resolution"]
    check_exist = cfg.get("check_exist", False)
    # Option to process a single file
    process_file = cfg.get("process_file", None)

    # Get PMW sources only
    sources, source_files, source_groups = list_tc_primed_sources(
        tc_primed_path,
        source_type="environmental",
    )
    era5_files = source_files.get("era5", [])

    # If process_file is specified, only process that file
    if process_file is not None:
        process_file_path = Path(process_file)
        if not process_file_path.is_absolute():
            process_file_path = tc_primed_path / process_file_path

        print(f"Processing only file: {process_file_path}")
        era5_files = [str(process_file_path)]

    source_dest_dir = dest_path / f"tc_primed_storm_metadata"
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
                        check_exist,
                        verbose=(i == 0),
                    )
                )
            for future in as_completed(futures):
                meta_chunk, discarded_chunk = future.result()
                samples_metadata.extend(meta_chunk)
                discarded += discarded_chunk
    else:
        for file in tqdm(era5_files, desc=f"Processing files"):
            file_times_metadata = process_storm_data_file(file, source_dest_dir, check_exist)
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
        print(f"Found no valid samples, removing directory.")
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
