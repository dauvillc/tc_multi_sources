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
import numpy as np
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
    regrid_to_grid,
)

# Local imports
from preproc.utils import list_tc_primed_overpass_files_by_sensat

# Dict of instrument -> {swath_containing_37GHz: [list_of_variables],
#                             swath_containing_89GHz: [list_of_variables], ...}
# or {swath: [list_of_variables]} if a single swath contains both 37GHz and 89GHz.
# Written from the TC-PRIMED documentation.
SENSOR_VARIABLES = {
    "AMSR2": {"37": ("S4", ["36.5H", "36.5V"]), "89": ("S5", ["A89.0H", "A89.0V"])},
    "AMSRE": {"37": ("S4", ["36.5H", "36.5V"]), "89": ("S5", ["A89.0H", "A89.0V"])},
    "GMI": {"37": ("S1", ["36.64H", "36.64V"]), "89": ("S1", ["89.0H", "89.0V"])},
    "SSMI": {"37": ("S1", ["37.0H", "37.0V"]), "89": ("S2", ["85.5H", "85.5V"])},
    "SSMIS": {"37": ("S2", ["37.0H", "37.0V"]), "89": ("S4", ["91.665H", "91.665V"])},
    "TMI": {"37": ("S2", ["37.0H", "37.0V"]), "89": ("S3", ["85.5H", "85.5V"])},
}
# Add the "TB" prefix to all variable names
for sensor in SENSOR_VARIABLES:
    for freq in SENSOR_VARIABLES[sensor]:
        swath, vars = SENSOR_VARIABLES[sensor][freq]
        SENSOR_VARIABLES[sensor][freq] = (swath, [f"TB_{var}" for var in vars])


def parse_frequency(pm_var_name):
    """Parse frequency from PMW variable name."""
    freq = pm_var_name.split("_")[1]
    for char in ["A", "B", "H", "V", "Q"]:
        freq = freq.replace(char, "")
    return float(freq)


def preprocess_pmw(ds):
    """Preprocess PMW data by standardizing dimensions and variables."""
    ds = ds.drop_vars(["ScanTime", "angle_bins", "x", "y"])
    # Make sure latitude and longitude are coordinates
    ds = ds.set_coords(["latitude", "longitude"])
    return ds


def initialize_pmw_metadata(sensat, ifovs, dest_dir):
    """Retrieves the list of all data and
    characteristic variables of a source; and writes the source metadata file.
    Args:
        sensat (str): Sensor / Satellite pair (e.g. "AMSR2_GCOMW1").
        ifovs (dict): Dictionary containing the IFOV values for each sensor/satellite and swath.
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """
    sensor = sensat.split("_")[0]
    if sensor not in SENSOR_VARIABLES:
        return False
    # The data variables are all the 37GHz and 89GHz variables.
    data_vars = SENSOR_VARIABLES[sensor]["37"][1] + SENSOR_VARIABLES[sensor]["89"][1]
    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": "tc_primed_pmw_" + sensat,
        "source_type": "pmw",
        "dim": 2,
        "data_vars": data_vars,
        "storm_metadata_vars": storm_vars,
    }

    # We'll now add to the source metadata the characteristic variables. For PMW
    # data, those are:
    # - the frequency of the sensor (one float value for each data variable).
    # - the IFOV (four float values for each data variable).
    # 1) Frequency: we'll extract it from the variable names.
    frequencies = {var: parse_frequency(var) for var in data_vars}
    source_metadata["charac_vars"] = {"frequency": frequencies}
    ifov_values = {}
    ifov_var_names = [
        "IFOV_nadir_along_track",
        "IFOV_nadir_across_track",
        "IFOV_edge_along_track",
        "IFOV_edge_across_track",
    ]
    if sensat not in ifovs:
        raise ValueError(f"Sensor {sensat} not found in IFOV file.")
    for freq, (swath, vars) in SENSOR_VARIABLES[sensor].items():
        if swath not in ifovs[sensat]:
            raise ValueError(f"Swath {swath} not found in IFOV file for sensor {sensat}.")
        ifov_values = ifovs[sensat][swath]
        # For each swath, the IFOV values are either a list or a dict.
        # - If a list, then the IFOV values are the same for all variables, so
        # we'll just repeat the list for each variable.
        if isinstance(ifov_values, list):
            ifov_values = {var: ifov_values for var in vars}
        # Now, ifov_values is always a dict {var_name: [ifov1, ifov2, ifov3, ifov4]}.
        for i, ifov_var in enumerate(ifov_var_names):
            source_metadata["charac_vars"][ifov_var] = {var: ifov_values[var][i] for var in vars}
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)
    return True


def process_pmw_file(file, sensat, dest_dir, check_older=None, to_regular_grid=False, ifovs=None):
    """Processes a single PMW file.
    Args:
        file (str): Path to the PMW file in netCDF4 format.
        sensat (str): Sensor / Satellite pair (e.g. "AMSR2_GCOMW1").
        dest_dir (Path): Destination directory.
        regridding_res (float): Resolution of the target grid, in degrees.
        check_older (timedelta or None): if a timedelta dt, checks if there is a pre-existing
            file younger than dt. If so, skips processing.
        to_regular_grid (bool): if True, regrids the data to a regular lat-lon grid.
            The 89GHz swath is first regridded using the resolution given in the IFOVs file,
            then the 37GHz swath is regridded to the same grid.
        ifovs (dict or None): Must be provided if to_regular_grid is True. Dict containing the
            IFOV values for each sensor/satellite and swath.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
    """
    with Dataset(file) as raw_sample:
        sensor = sensat.split("_")[0]
        swath_37, data_vars_37 = SENSOR_VARIABLES[sensor]["37"]
        swath_89, data_vars_89 = SENSOR_VARIABLES[sensor]["89"]

        # We'll first process the 89GHz swath (second one), as it will serve as
        # the base grid for regridding the 37GHz swath (first one).
        # We'll begin by extracting the sample metadata, which is common to both swaths.
        ds89 = raw_sample["passive_microwave"][swath_89]
        ds89 = xr.open_dataset(NetCDF4DataStore(ds89), decode_times=False)
        ds89 = preprocess_pmw(ds89)

        # Assemble the sample's metadata
        raw_overpass_meta = raw_sample["overpass_metadata"]
        season = raw_overpass_meta["season"][0].item()
        basin = raw_overpass_meta["basin"][0]
        storm_number = raw_overpass_meta["cyclone_number"][-1].item()
        sid = f"{season}{basin}{storm_number}"  # Unique storm identifier
        time = pd.to_datetime(raw_overpass_meta["time"][0], origin="unix", unit="s")
        overpass_storm_metadata = raw_sample["overpass_storm_metadata"]
        storm_lat = overpass_storm_metadata["storm_latitude"][0].item()
        storm_lon = overpass_storm_metadata["storm_longitude"][0].item()
        storm_lon = (storm_lon + 180) % 360 - 180  # Standardize longitude values
        intensity = overpass_storm_metadata["intensity"][0].item()

        sample_metadata = {
            "source_name": "tc_primed_pmw_" + sensat,
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

        # === 89GHZ PROCESSING ===
        # Extract data variables
        sensor = sensat.split("_")[0]
        # Only keep the relevant variables
        ds89 = ds89[data_vars_89 + ["latitude", "longitude"]]
        # If any variable is fully missing data, skip
        if any(ds89[var].isnull().all() for var in data_vars_89):
            return None
        # Standardize longitude values to [-180, 180]
        ds89["longitude"] = (ds89["longitude"] + 180) % 360 - 180

        # OPTIONAL REGRIDDING TO REGULAR GRID
        if to_regular_grid:
            if ifovs is None:
                raise ValueError("ifovs must be provided if to_regular_grid is True.")
            # Get the IFOV values for the 89GHz swath (as nadir along and across track)
            ifovs_89 = ifovs[sensat][swath_89]
            # ifovs_89 is either a list [nadir_along, nadir_across, edge_along, edge_across]
            # or a dict {var_name: [nadir_along, nadir_across, edge_along, edge_across]}.
            if isinstance(ifovs_89, dict):
                # We'll just take the IFOVs of the first variable
                first_var = data_vars_89[0]
                ifovs_89 = ifovs_89[first_var]
            ifov_nadir_along, ifov_nadir_across = ifovs_89[0], ifovs_89[1]
            # Estimate regridding resolution as the max IFOV
            regridding_res_km = max(ifov_nadir_across, ifov_nadir_along)

            # Convert km to degrees for latitude (constant)
            regridding_res_lat = regridding_res_km / 111.32

            # Create latitude grid first
            lat89 = ds89["latitude"].values
            min_lat, max_lat = lat89.min(), lat89.max()
            new_lats = np.arange(min_lat, max_lat, regridding_res_lat)

            # For longitude, compute resolution at each latitude
            lon89 = ds89["longitude"].values
            min_lon, max_lon = lon89.min(), lon89.max()

            # Use latitude-dependent longitude spacing
            # Calculate resolution at the center of the latitude range for a uniform grid
            center_lat = (min_lat + max_lat) / 2
            regridding_res_lon = regridding_res_km / (111.32 * np.cos(np.radians(center_lat)))
            new_lons = np.arange(min_lon, max_lon, regridding_res_lon)
            new_lons = (new_lons + 180) % 360 - 180  # Standardize to [-180, 180]
            grid_lons, grid_lats = np.meshgrid(new_lons, new_lats)
            # Regrid the 89GHz data to the regular grid
            try:
                ds89 = regrid_to_grid(ds89, grid_lats, grid_lons)
                # Check if any variable is fully null after regridding
                for variable in ds89.variables:
                    if ds89[variable].isnull().all():
                        warnings.warn(f"Variable {variable} is null for sample {file}.")
            except ResamplingError as e:
                raise RuntimeError(f"Resampling error for sample {file}: {e}")

        # === 37GHZ PROCESSING ===
        ds37 = raw_sample["passive_microwave"][swath_37]
        ds37 = xr.open_dataset(NetCDF4DataStore(ds37), decode_times=False)
        ds37 = preprocess_pmw(ds37)
        # Only keep the relevant variables
        ds37 = ds37[data_vars_37 + ["latitude", "longitude"]]
        # If any variable is fully missing data, skip
        if any(ds37[var].isnull().all() for var in data_vars_37):
            return None
        # Standardize longitude values to [-180, 180]
        ds37["longitude"] = (ds37["longitude"] + 180) % 360 - 180
        # Regrid to the 89GHz grid
        try:
            ds37 = regrid_to_grid(ds37, ds89["latitude"].values, ds89["longitude"].values)
            # Check if any variable is fully null after regridding. If so, skip the sample.
            for variable in ds37.variables:
                if ds37[variable].isnull().all():
                    warnings.warn(f"Variable {variable} is null for sample {file}.")
                    return None
        except ResamplingError as e:
            raise RuntimeError(f"Resampling error for sample {file}: {e}")
        # Add the regridded 37GHz variables to the 89GHz dataset
        ds = xr.merge([ds89, ds37], compat="override")

        # Compute the land-sea mask
        try:
            land_mask = globe.is_land(
                ds["latitude"].values,
                ds["longitude"].values,
            )
        except ValueError as e:
            # Add info to the error message about the sample
            error_msg = f"Error computing land mask for sample {file}: {e}"
            error_msg += (
                f"\nMin lat: {ds['latitude'].values.min()}, Max lat: {ds['latitude'].values.max()}"
            )
            error_msg += f"\nMin lon: {ds['longitude'].values.min()}, Max lon: {ds['longitude'].values.max()}"
            raise ValueError(error_msg)
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
        ds = ds[
            data_vars_89 + data_vars_37 + ["latitude", "longitude", "land_mask", "dist_to_center"]
        ]
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

    check_older = cfg.get("check_older", None)
    check_older = pd.to_timedelta(check_older) if check_older is not None else None

    use_regular_grid = cfg.get("use_regular_grid_pmw", False)

    # Read the IFOV values file
    with open(ifovs_path, "r") as f:
        ifovs = yaml.safe_load(f)

    # Retrieve all TC-PRIMED overpass files as a dict {sen_sat: <file_list>}
    pmw_files = list_tc_primed_overpass_files_by_sensat(tc_primed_path)

    # Process each PMW source. "sensat" stands for Sensor/Satellite (e.g. "AMSR2_GCOMW1").
    for sensat in pmw_files:
        source_dest_dir = dest_path / f"tc_primed_pmw_{sensat}"
        # Create destination directory
        source_dest_dir.mkdir(parents=True, exist_ok=True)

        if not initialize_pmw_metadata(sensat, ifovs, source_dest_dir):
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
            for file in tqdm(pmw_files[sensat], desc=f"Processing {sensat}"):
                sample_metadata = process_pmw_file(
                    file,
                    sensat,
                    source_dest_dir,
                    check_older,
                    to_regular_grid=use_regular_grid,
                    ifovs=ifovs,
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
                    repeat(source_dest_dir),
                    repeat(check_older),
                    repeat(use_regular_grid),
                    repeat(ifovs),
                    chunksize=chunksize,
                )
                for sample_metadata in tqdm(
                    iterator,
                    desc=f"Processing {sensat}",
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
            print(f"Found no valid samples for {sensat}, removing directory.")
        else:
            if discarded > 0:
                percent = discarded / len(pmw_files[sensat]) * 100
                print(f"Discarded {discarded} samples for {sensat} ({percent:.2f}%)")
            # Concatenate the samples metadata into a DataFrame and save it
            samples_metadata = pd.DataFrame(samples_metadata)
            samples_metadata_path = source_dest_dir / "samples_metadata.csv"
            samples_metadata.to_csv(samples_metadata_path, index=False)


if __name__ == "__main__":
    main()
