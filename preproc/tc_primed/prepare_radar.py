#!/usr/bin/env python3

"""
Extracts the radar data from the TC-PRIMED overpass files.
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

SENSAT_VARIABLES = {
    "GMI_GPM": {
        "KuGMI": ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma", "mainprecipitationType"]
    },
    "TMI_TRMM": {
        "KuTMI": ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma", "mainprecipitationType"]
    },
}


def preprocess_radar(ds):
    """Preprocess radar data by standardizing dimensions and variables."""
    ds = ds.drop_vars(["ScanTime", "angle_bins", "x", "y"])
    return ds


def initialize_radar_metadata(sensat, swath, ifovs_path, dest_dir):
    """Retrieves the list of all data and
    characteristic variables of a source; and writes the source metadata file.
    Args:
        sensat (str): Sensor / Satellite pair (e.g. "PR_TRMM")
        swath (str): Swath name (e.g. "KuTMI").
        ifovs_path (Path): Path to the IFOVs YAML file.
        dest_dir (Path): Destination directory.
    Returns:
        bool: True if the metadata was successfully written, False otherwise.
    """
    data_vars = SENSAT_VARIABLES[sensat][swath]
    # Variables that give information about the storm of the sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Dict that contains the source's metadata.
    source_metadata = {
        "source_name": "tc_primed_radar_" + sensat + "_" + swath,
        "source_type": "radar",
        "dim": 2,
        "data_vars": data_vars,
        "storm_metadata_vars": storm_vars,
    }

    # We'll now add to the source metadata the characteristic variables. For radar data,
    # this will only be the IFOV as the sensors are equivalent otherwise.
    # We'll get the IFOV values from the IFOV file (which needs to be
    # manually filled based on the documentation).
    source_metadata["charac_vars"] = {}
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


def process_overpass_file(
    file, sensat, swath, dest_dir, rain_rate_criteria, regridding_res, check_older=None
):
    """Processes a single overpass file to extract the radar data if it is available.
    Args:
        file (str): Path to the overpass file in netCDF4 format.
        sensat (str): Sensor / Satellite pair (e.g. "PR_TRMM")
        swath (str): Swath name (e.g. "KuTMI").
        dest_dir (Path): Destination directory.
        rain_rate_criteria (list of pairs of float): List of pairs (number of pixels, min rain rate)
            such that only radar images with at least that many pixels above that rain rate
            will be kept. Including multiple pairs will act as a logical OR.
        regridding_res (float): Resolution of the target grid, in degrees.
        check_older (timedelta or None): if a timedelta dt, checks if there is a pre-existing
            file younger than dt. If so, skips processing.
    Returns:
        dict or None: Sample metadata, or None if the sample is discarded.
    """
    with Dataset(file) as raw_sample:
        ds = raw_sample["radar_radiometer"]
        # Look at the radar data availability flag
        if ds["availability_flag"][:].item() == 0:
            return None

        # Open in xarray and preprocess dataset
        ds = xr.open_dataset(NetCDF4DataStore(ds[swath]), decode_times=False)
        ds = preprocess_radar(ds)

        # Extract data variables
        data_vars = SENSAT_VARIABLES[sensat][swath]
        ds = ds[data_vars + ["latitude", "longitude"]]
        # If any variable is fully missing data, skip
        if any(ds[var].isnull().all() for var in data_vars):
            return None
        # Check if the sample meets the rain rate criteria
        ds_rain = ds["nearSurfPrecipTotRate"].values
        meets_criteria = False
        for num_pixels, min_rain_rate in rain_rate_criteria:
            if (ds_rain > min_rain_rate).sum() >= num_pixels:
                meets_criteria = True
                break
        if not meets_criteria:
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
            "source_name": "tc_primed_radar_" + sensat + "_" + swath,
            "source_type": "radar",
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
    """Main function to process the radar data."""
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup paths
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    ifovs_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed_ifovs.yaml"
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"
    # Resolution of the target grid, in degrees
    regridding_res = cfg["regridding_resolution_radar"]
    # Rain rate criteria for radar pre-selection
    rain_rate_criteria = cfg["radar_rain_rate_criteria"]

    check_older = cfg.get("check_older", None)
    check_older = pd.to_timedelta(check_older) if check_older is not None else None

    # Retrieve all TC-PRIMED overpass files as a dict {sen_sat: <file_list>}
    overpass_files = list_tc_primed_overpass_files_by_sensat(tc_primed_path)
    overpass_files = {
        sensat: files for sensat, files in overpass_files.items() if sensat in SENSAT_VARIABLES
    }

    # Process each source. "sensat" stands for Sensor/Satellite (e.g. "PR_TRMM").
    for sensat in overpass_files:
        for swath in SENSAT_VARIABLES[sensat].keys():
            source_dest_dir = dest_path / f"tc_primed_radar_{sensat}_{swath}"
            # Create destination directory
            source_dest_dir.mkdir(parents=True, exist_ok=True)

            if not initialize_radar_metadata(sensat, swath, ifovs_path, source_dest_dir):
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
                for file in tqdm(
                    overpass_files[sensat], desc=f"Processing {sensat} swath {swath}"
                ):
                    sample_metadata = process_overpass_file(
                        file,
                        sensat,
                        swath,
                        source_dest_dir,
                        rain_rate_criteria,
                        regridding_res,
                        check_older,
                    )
                    if sample_metadata is None:
                        discarded += 1
                    else:
                        samples_metadata.append(sample_metadata)
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    iterator = executor.map(
                        process_overpass_file,
                        overpass_files[sensat],
                        repeat(sensat),
                        repeat(swath),
                        repeat(source_dest_dir),
                        repeat(rain_rate_criteria),
                        repeat(regridding_res),
                        repeat(check_older),
                        chunksize=chunksize,
                    )
                    for sample_metadata in tqdm(
                        iterator,
                        desc=f"Processing {sensat} swath {swath}",
                        total=len(overpass_files[sensat]),
                    ):
                        if sample_metadata is None:
                            discarded += 1
                        else:
                            samples_metadata.append(sample_metadata)

            # If all files were discarded, remove all files inside the directory
            # and remove the directory itself.
            if discarded == len(overpass_files[sensat]):
                for file in source_dest_dir.iterdir():
                    file.unlink()
                source_dest_dir.rmdir()
                print(f"Found no valid samples for {sensat}/{swath}, removing directory.")
            else:
                if discarded > 0:
                    percent = discarded / len(overpass_files[sensat]) * 100
                    print(f"Discarded {discarded} samples for {sensat}/{swath} ({percent:.2f}%)")
                # Concatenate the samples metadata into a DataFrame and save it
                samples_metadata = pd.DataFrame(samples_metadata)
                samples_metadata_path = source_dest_dir / "samples_metadata.csv"
                samples_metadata.to_csv(samples_metadata_path, index=False)


if __name__ == "__main__":
    main()
