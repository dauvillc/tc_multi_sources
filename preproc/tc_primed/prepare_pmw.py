#!/usr/bin/env python3

"""
Prepare passive microwave (PMW) data from TC-PRIMED dataset.
This script processes PMW data and saves it in a standardized format with metadata.
"""

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import hydra
import pandas as pd
import xarray as xr
import yaml
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
from preproc.utils import list_tc_primed_overpass_files_by_sensat

CROSS_TRACK_INSTRUMENTS = ["AMSU", "ATMS", "MHS"]
# Dict of instrument -> swath -> frequencies to process. Other ones will be ignored.
# Written from the TC-PRIMED documentation.
SENSOR_VARIABLES = {
    "AMSR2": {"S4": ["36.5H", "36.5V"], "S5": ["A89.0H", "A89.0V"], "S6": ["B89.0H", "B89.0V"]},
    "AMSRE": {"S4": ["36.5H", "36.5V"], "S5": ["A89.0H", "A89.0V"], "S6": ["B89.0H", "B89.0V"]},
    "AMSUB": {"S1": ["89.0_0.9QV"]},
    "ATMS": {"S3": ["88.2QV"]},
    "GMI": {"S1": ["36.64H", "36.64V", "89.0H", "89.0V"]},
    "MHS": {"S1": ["89.0V"]},
    "SSMI": {"S1": ["37.0H", "37.0V"], "S2": ["85.5H", "85.5V"]},
    "SSMIS": {"S2": ["37.0H", "37.0V"], "S4": ["91.665H", "91.665V"]},
    "TMI": {"S2": ["37.0H", "37.0V"], "S3": ["85.5H", "85.5V"]},
}
# Add the "TB_" prefix to every variable name.
SENSOR_VARIABLES = {
    sensor: {swath: ["TB_" + var for var in vars] for swath, vars in swaths.items()}
    for sensor, swaths in SENSOR_VARIABLES.items()
}


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


def initialize_pmw_metadata(sensat, swath, ifovs_path, dest_dir):
    """Retrieves the list of all data and
    characteristic variables of a source; and writes the source metadata file.
    Args:
        sensat (str): Sensor / Satellite pair (e.g. "AMSR2_GCOMW1").
        swath (str): Swath name (e.g. "S4").
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """
    sensor = sensat.split("_")[0]
    data_vars = SENSOR_VARIABLES[sensor][swath]
    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": "tc_primed_pmw_" + sensat + "_" + swath,
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
    # Check if the IFOV values are available for the instrument and swath.
    if sensat not in ifovs or swath not in ifovs[sensat]:
        print(f"IFOV values not found for {sensat}.")
        return False
    ifov_values = ifovs[sensat][swath]
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


def process_pmw_file(file, sensat, swath, dest_dir, regridding_res, check_older=None):
    """Processes a single PMW file.
    Args:
        file (str): Path to the PMW file in netCDF4 format.
        sensat (str): Sensor / Satellite pair (e.g. "AMSR2_GCOMW1").
        swath (str): Swath name (e.g. "S4").
        dest_dir (Path): Destination directory.
        regridding_res (float): Resolution of the target grid, in degrees.
        check_older (timedelta or None): if a timedelta dt, checks if there is a pre-existing
            file younger than dt. If so, skips processing.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
    """
    with Dataset(file) as raw_sample:
        ds = raw_sample["passive_microwave"][swath]

        # Open in xarray and preprocess dataset
        ds = xr.open_dataset(NetCDF4DataStore(ds), decode_times=False)
        ds = preprocess_pmw(ds)

        # Extract data variables
        sensor = sensat.split("_")[0]
        data_vars = SENSOR_VARIABLES[sensor][swath]
        # If any variable is fully missing data, skip
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
        # Intensity
        intensity = overpass_storm_metadata["intensity"][0].item()

        sample_metadata = {
            "source_name": "tc_primed_pmw_" + sensat + "_" + swath,
            "source_type": "pmw",
            "sid": sid,
            "time": time,
            "season": season,
            "storm_latitude": storm_lat,
            "storm_longitude": storm_lon,
            "intensity": intensity,
            "basin": basin,
            "dim": 2,  # Spatial dimensionality
        }
        dest_file = dest_dir / f"{sid}_{time.strftime('%Y%m%dT%H%M%S')}.nc"
        sample_metadata["data_path"] = dest_file

        # Check if the file already exists and is younger than the max timedelta
        if dest_file.exists() and check_older is not None:
            mtime = pd.to_datetime(dest_file.stat().st_mtime, unit="s")
            if pd.Timestamp.now() - mtime < check_older:
                return sample_metadata

        # Standardize longitude values to [-180, 180]
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180

        # Regrid from satellite geometry to a regular grid
        try:
            # For cross-track instruments, we'll crop the borders as the pixels
            # at the edges have an unusable IFOV.
            sensor = sensat.split("_")[0]
            if sensor in CROSS_TRACK_INSTRUMENTS:
                ds = ds.isel(pixel=slice(10, -10))
            # Regrid from satellite geometry to a regular lat/lon grid
            ds = regrid(ds, regridding_res)
            # Check if any variable is fully null after regridding. If so, skip the sample.
            for variable in ds.variables:
                if ds[variable].isnull().all():
                    warnings.warn(f"Variable {variable} is null for sample {file}.")
                    return None
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
        ds = ds[data_vars + ["latitude", "longitude", "land_mask", "dist_to_center"]]
        ds.to_netcdf(dest_file)

        return sample_metadata


@hydra.main(config_path="../../conf/", config_name="preproc", version_base=None)
def main(cfg):
    """Main function to process PMW data."""
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup paths
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    ifovs_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed_ifovs.yaml"
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"
    # Resolution of the target grid, in degrees
    regridding_res = cfg["regridding_resolution"]

    check_older = cfg.get("check_older", None)
    check_older = pd.to_timedelta(check_older) if check_older is not None else None

    # Retrieve all TC-PRIMED overpass files as a dict {sen_sat: <file_list>}
    pmw_files = list_tc_primed_overpass_files_by_sensat(tc_primed_path)

    # Process each PMW source. "sensat" stands for Sensor/Satellite (e.g. "AMSR2_GCOMW1").
    for sensat in pmw_files:
        sensor = sensat.split("_")[0]
        for swath in SENSOR_VARIABLES[sensor].keys():
            source_dest_dir = dest_path / f"tc_primed_pmw_{sensat}_{swath}"
            # Create destination directory
            source_dest_dir.mkdir(parents=True, exist_ok=True)

            if not initialize_pmw_metadata(sensat, swath, ifovs_path, source_dest_dir):
                # Remove directory if metadata could not be initialized
                source_dest_dir.rmdir()
                print(f"Skipping {sensat}")
                continue

            # Process all files, and keep count of discarded files and save the metadata of
            # each sample.
            discarded, samples_metadata = 0, []
            num_workers = cfg.get("num_workers", 1)
            chunksize = cfg.get("chunksize", 64)

            if num_workers <= 1:
                for file in tqdm(pmw_files[sensat], desc=f"Processing {sensat} swath {swath}"):
                    sample_metadata = process_pmw_file(
                        file, sensat, swath, source_dest_dir, regridding_res, check_older
                    )
                    if sample_metadata is None:
                        discarded += 1
                    else:
                        samples_metadata.append(sample_metadata)
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    iterator = executor.map(
                        process_pmw_file,
                        pmw_files[sensat],
                        repeat(sensat),
                        repeat(swath),
                        repeat(source_dest_dir),
                        repeat(regridding_res),
                        repeat(check_older),
                        chunksize=chunksize,
                    )
                    for sample_metadata in tqdm(
                        iterator,
                        desc=f"Processing {sensat} swath {swath}",
                        total=len(pmw_files[sensat]),
                    ):
                        if sample_metadata is None:
                            discarded += 1
                        else:
                            samples_metadata.append(sample_metadata)

            # If all files were discarded, remove all files inside the directory
            # and remove the directory itself.
            if discarded == len(pmw_files[sensat]):
                for file in source_dest_dir.iterdir():
                    file.unlink()
                source_dest_dir.rmdir()
                print(f"Found no valid samples for {sensat}/{swath}, removing directory.")
            else:
                if discarded > 0:
                    percent = discarded / len(pmw_files[sensat]) * 100
                    print(f"Discarded {discarded} samples for {sensat}/{swath} ({percent:.2f}%)")
                # Concatenate the samples metadata into a DataFrame and save it
                samples_metadata = pd.DataFrame(samples_metadata)
                samples_metadata_path = source_dest_dir / "samples_metadata.csv"
                samples_metadata.to_csv(samples_metadata_path, index=False)


if __name__ == "__main__":
    main()
