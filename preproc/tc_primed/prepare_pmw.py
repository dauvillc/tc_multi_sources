#!/usr/bin/env python3

"""
Prepare passive microwave (PMW) data from TC-PRIMED dataset.
This script processes PMW data and saves it in a standardized format with metadata.
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
from multi_sources.data_processing.grid_functions import (
    regrid,
    grid_distance_to_point,
    ResamplingError,
)


def parse_frequency(pm_var_name):
    """Parse frequency from PMW variable name."""
    freq = pm_var_name.split("_")[1]
    for char in ["A", "B", "H", "V", "Q"]:
        freq = freq.replace(char, "")
    return float(freq)


def preprocess_pmw(ds):
    """Preprocess PMW data by standardizing dimensions and variables."""
    ds = ds.drop_vars(["ScanTime", "angle_bins", "x", "y"])
    return ds


def initialize_pmw_metadata(ds, source, ifovs_path, dest_dir):
    """Given a PMW dataset, retrieves the list of all data and
    characteristic variables of a source; and writes the source metadata file.
    Args:
        ds (xr.Dataset): any sample from the source.
        source (str): Source name.
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """
    # Retrieve the list of data variables from the sample.
    data_vars = [var for var in ds.data_vars if var.startswith("TB")]

    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": "tc_primed_" + source,
        "source_type": "pmw",
        "dim": 2,
        "data_vars": data_vars,
        "storm_metadata_vars": storm_vars,
        "is_on_regular_grid": False,
    }

    # We'll now add to the source metadata the characteristic variables. For PMW
    # data, those are:
    # - the frequency of the sensor (one float value for each data variable).
    # - the IFOV (four float values for each data variable).
    # 1) Frequency: we'll extract it from the variable names.
    frequencies = {var: parse_frequency(var) for var in data_vars}
    source_metadata["charac_vars"] = {"frequency": frequencies}
    # 2) IFOV: we'll get the IFOV values from the IFOV file (which needs to be
    # manually filled based on the documentation).
    with open(ifovs_path, "r") as f:
        ifovs = yaml.safe_load(f)
    # Extract the name of the instrument and swath from the source name.
    source_parts = source.split("_")
    instrument = "_".join(source_parts[1:-1])
    swath = source_parts[-1]
    # Check if the IFOV values are available for the instrument and swath.
    if instrument not in ifovs or swath not in ifovs[instrument]:
        print(f"IFOV values not found for {source}.")
        return False
    # Get the IFOV values for the instrument and swath.
    ifov_values = ifovs[instrument][swath]
    # Now, the IFOV values are either a list or a dict.
    # - If a list, then the IFOV values are the same for all variables, so
    # we'll just repeat the list for each variable.
    if isinstance(ifov_values, list):
        ifov_values = {var: ifov_values for var in data_vars}
    # - If a dict, then the IFOV values are different for each variable.
    ifov_var_names = [
        "IFOV_nadir_along_track",
        "IFOV_nadir_across_track",
        "IFOV_edge_along_track",
        "IFOV_edge_across_track",
    ]
    for i, ifov_var in enumerate(ifov_var_names):
        source_metadata["charac_vars"][ifov_var] = {var: ifov_values[var][i] for var in data_vars}
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)
    return True


def process_pmw_file(file, source, source_groups, dest_dir, regridding_res):
    """Processes a single PMW file.
    Args:
        file (str): Path to the PMW file.
        source (str): Source name.
        source_groups (list): List of groups in the source file.
        dest_dir (Path): Destination directory.
        regridding_res (float): Resolution of the target grid, in degrees.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
    """
    with Dataset(file) as raw_sample:
        # Access data group
        ds = raw_sample
        for group in source_groups:
            if group in ds.groups:
                ds = ds[group]

        # Open and preprocess dataset
        ds = xr.open_dataset(NetCDF4DataStore(ds), decode_times=False)
        ds = preprocess_pmw(ds)

        # Extract data variables
        data_vars = [var for var in ds.data_vars if var.startswith("TB")]
        # Check for missing data
        if any(ds[var].isnull().all() for var in data_vars):
            return None

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
            "source_name": "tc_primed_" + source,
            "source_type": "pmw",
            "sid": sid,
            "time": time,
            "season": season,
            "storm_latitude": storm_lat,
            "storm_longitude": storm_lon,
            "basin": basin,
            "dim": 2,  # Spatial dimensionality
        }

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
        new_ds = ds[data_vars + ["latitude", "longitude", "land_mask", "dist_to_center"]]
        dest_file = dest_dir / f"{sid}_{time.strftime('%Y%m%dT%H%M%S')}.nc"
        new_ds.to_netcdf(dest_file)
        sample_metadata["data_path"] = dest_file

        return sample_metadata


# Helper function to process a chunk of files in parallel.
def process_chunk(file_list, source, pmw_groups, source_dest_dir, regridding_res, verbose=False):
    local_metadata = []
    discarded = 0

    iterator = tqdm(file_list, desc="Processing chunk") if verbose else file_list
    for file in iterator:
        meta = process_pmw_file(file, source, pmw_groups, source_dest_dir, regridding_res)
        if meta is None:
            discarded += 1
        else:
            local_metadata.append(meta)
    return local_metadata, discarded


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(cfg):
    """Main function to process PMW data."""
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup paths
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    ifovs_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed_ifovs.yaml"
    with open(ifovs_path, "r") as f:
        ifovs = yaml.safe_load(f)
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"
    # Resolution of the target grid, in degrees
    regridding_res = cfg["regridding_resolution"]

    # Get PMW sources only
    sources, source_files, source_groups = list_tc_primed_sources(
        tc_primed_path, source_type="satellite"
    )
    pmw_files = {s: files for s, files in source_files.items() if s.startswith("pmw_")}
    pmw_groups = {s: source_groups[s] for s in pmw_files.keys()}

    # Process only a specific source if provided in cfg
    if "process_only" in cfg:
        process_source = cfg["process_only"]
        if process_source in pmw_files:
            pmw_files = {process_source: pmw_files[process_source]}
            pmw_groups = {process_source: pmw_groups[process_source]}
        else:
            print(
                f"Source {process_source} not found in available sources: {list(pmw_files.keys())}"
            )
            return

    # Process each PMW source
    for source in pmw_files:
        print(f"Processing {source}")
        source_dest_dir = dest_path / f"tc_primed_{source}"
        # Create destination directory
        source_dest_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata using first file
        with Dataset(pmw_files[source][0]) as raw_sample:
            # Recursively access groups associated with the source
            ds = raw_sample
            for group in pmw_groups[source]:
                ds = ds[group]
            ds = xr.open_dataset(NetCDF4DataStore(ds), decode_times=False)
            if not initialize_pmw_metadata(ds, source, ifovs_path, source_dest_dir):
                # Remove directory if metadata could not be initialized
                source_dest_dir.rmdir()
                print(f"Skipping {source}")
                continue

        # Process all files, and keep count of discarded files and save the metadata of
        # each sample.
        discarded, samples_metadata = 0, []
        num_workers = cfg.get("num_workers", 1)

        if num_workers > 1:
            chunks = np.array_split(pmw_files[source], num_workers)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    futures.append(
                        executor.submit(
                            process_chunk,
                            list(chunk),
                            source,
                            pmw_groups[source],
                            source_dest_dir,
                            regridding_res,
                            verbose=(i == 0),
                        )
                    )
                for future in as_completed(futures):
                    meta_chunk, discarded_chunk = future.result()
                    samples_metadata.extend(meta_chunk)
                    discarded += discarded_chunk
        else:
            for file in tqdm(pmw_files[source], desc=f"Processing {source}"):
                sample_metadata = process_pmw_file(
                    file, source, pmw_groups[source], source_dest_dir, regridding_res
                )
                if sample_metadata is None:
                    discarded += 1
                else:
                    samples_metadata.append(sample_metadata)

        # If all files were discarded, remove all files inside the directory
        # and remove the directory itself.
        if discarded == len(pmw_files[source]):
            for file in source_dest_dir.iterdir():
                file.unlink()
            source_dest_dir.rmdir()
            print(f"Found no valid samples for {source}, removing directory.")
        else:
            if discarded > 0:
                percent = discarded / len(pmw_files[source]) * 100
                print(f"Discarded {discarded} samples for {source} ({percent:.2f}%)")
            # Concatenate the samples metadata into a DataFrame and save it
            samples_metadata = pd.DataFrame(samples_metadata)
            samples_metadata_path = source_dest_dir / "samples_metadata.csv"
            samples_metadata.to_csv(samples_metadata_path, index=False)


if __name__ == "__main__":
    main()
