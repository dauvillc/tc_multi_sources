#!/usr/bin/env python3

"""
Prepares the infrared data for the TC Primed dataset.
"""

import json
import warnings
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
    ResamplingError,
    grid_distance_to_point,
    regrid,
)

# Local imports
from preproc.utils import list_tc_primed_sources


def initialize_metadata(dest_dir):
    """Initializes the metadata for the infrared data.
    Args:
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """

    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": "tc_primed_infrared",
        "source_type": "infrared",
        "dim": 2,
        "data_vars": "IRWIN",
        "storm_metadata_vars": storm_vars,
        "is_on_regular_grid": True,
        "charac_vars": {},
    }
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)
    return True


def process_ir_file(file, dest_dir, regridding_res, check_exist=False):
    """Processes a single infrared file.s
    Args:
        file (str): Path to the input file.
        dest_dir (Path): Destination directory.
        regridding_res (float): Resolution of the target grid, in degrees.
        check_exist (bool): Flag to check if the file already exists.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
    """
    with Dataset(file) as raw_sample:
        # Access data group
        ds = raw_sample["infrared"]
        # Check the infrared availability flag. We only keep the samples
        # for which it is set to 1 (0 means unavailable, 2 means lower-res HURSAT data).
        if ds["infrared_availability_flag"][0].item() != 1:
            return None

        # Open the dataset using xarray and select the brightness temperature variable
        ds = xr.open_dataset(NetCDF4DataStore(ds), decode_times=False)
        ds = ds[["IRWIN", "latitude", "longitude"]]

        # Assemble the sample's metadata
        raw_overpass_meta = raw_sample["overpass_metadata"]
        season = raw_overpass_meta["season"][0].item()
        basin = raw_overpass_meta["basin"][0]
        storm_number = raw_overpass_meta["cyclone_number"][-1].item()
        sid = f"{season}{basin}{storm_number}"  # Unique storm identifier
        time = pd.to_datetime(raw_overpass_meta["time"][0], origin="unix", unit="s")
        # Storm latitude and longitude
        overpass_storm_metadata = raw_sample["overpass_storm_metadata"]
        storm_lat = overpass_storm_metadata["storm_latitude"][0].item()
        storm_lon = overpass_storm_metadata["storm_longitude"][0].item()
        storm_lon = (storm_lon + 180) % 360 - 180  # Standardize longitude values

        sample_metadata = {
            "source_name": "tc_primed_infrared",
            "source_type": "infrared",
            "sid": sid,
            "time": time,
            "season": season,
            "storm_latitude": storm_lat,
            "storm_longitude": storm_lon,
            "basin": basin,
            "dim": 2,  # Spatial dimensionality
        }
        dest_file = dest_dir / f"{sid}_{time.strftime('%Y%m%dT%H%M%S')}.nc"
        sample_metadata["data_path"] = dest_file

        # Check if the file already exists. If it does, skip processing.
        if check_exist and dest_file.exists():
            return sample_metadata

        # Standardize longitude values to [-180, 180]
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180

        # Regrid from satellite geometry to a regular grid
        try:
            ds = regrid(ds, regridding_res)
            # Check if any variable is fully null after regridding
            for variable in ds.variables:
                if ds[variable].isnull().all():
                    warnings.warn(f"Variable {variable} is null for sample {file}.")

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

        except ResamplingError as e:
            raise RuntimeError(f"Resampling error for sample {file}: {e}")

        # Save processed data in the netCDF format
        ds = ds[["IRWIN", "latitude", "longitude", "land_mask", "dist_to_center"]]
        ds.to_netcdf(dest_file)

        return sample_metadata


# Helper function to process a chunk of files in parallel.
def process_chunk(file_list, source_dest_dir, regridding_res, check_exist, verbose=False):
    local_metadata = []
    discarded = 0

    iterator = tqdm(file_list, desc="Processing chunk") if verbose else file_list
    for file in iterator:
        meta = process_ir_file(file, source_dest_dir, regridding_res, check_exist)
        if meta is None:
            discarded += 1
        else:
            local_metadata.append(meta)
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

    # Get PMW sources only
    sources, source_files, source_groups = list_tc_primed_sources(
        tc_primed_path, source_type="satellite"
    )
    ir_files = source_files["infrared"]

    source_dest_dir = dest_path / "tc_primed_infrared"
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
        chunks = np.array_split(ir_files, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(
                        process_chunk,
                        list(chunk),
                        source_dest_dir,
                        regridding_res,
                        check_exist,
                        verbose=(i == 0),
                    )
                )
            for future in as_completed(futures):
                meta_chunk, discarded_chunk = future.result()
                samples_metadata.extend(meta_chunk)
                discarded += discarded_chunk
    else:
        for file in tqdm(ir_files, desc="Processing files"):
            sample_metadata = process_ir_file(file, source_dest_dir, regridding_res, check_exist)
            if sample_metadata is None:
                discarded += 1
            else:
                samples_metadata.append(sample_metadata)

    # If all files were discarded, remove all files inside the directory
    # and remove the directory itself.
    if discarded == len(ir_files):
        for file in source_dest_dir.iterdir():
            file.unlink()
        source_dest_dir.rmdir()
        print("Found no valid samples, removing directory.")
    else:
        if discarded > 0:
            percent = discarded / len(ir_files) * 100
            print(f"Discarded {discarded} samples ({percent:.2f}%)")
        # Concatenate the samples metadata into a DataFrame and save it
        samples_metadata = pd.DataFrame(samples_metadata)
        samples_metadata_path = source_dest_dir / "samples_metadata.csv"
        samples_metadata.to_csv(samples_metadata_path, index=False)


if __name__ == "__main__":
    main()
