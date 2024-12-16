"""Usage: python prepare_tc_primed_era5.py (meant to be used with hydra).
Formats the TC-Primed ERA5 data into the common preprocessing format.
"""

import hydra
import pandas as pd
import json
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from netCDF4 import Dataset
from preproc.utils import list_tc_primed_sources
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def process_file(file, dest_dir, data_vars, source_name):
    sample_metadata_rows = []
    discarded_count = 0
    with Dataset(file) as raw_sample:
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(raw_sample["rectilinear"]))
        times = ds["time"].values

        # Get storm metadata
        storm_meta = raw_sample["storm_metadata"]
        season = storm_meta["season"][0].item()
        basin = storm_meta["basin"][0]
        storm_number = storm_meta["cyclone_number"][-1].item()
        sid = f"{season}{basin}{storm_number}"

        # Process each timestep
        for time_idx, time in enumerate(times):
            # Get data for this timestep
            timestep_ds = ds.isel(time=time_idx)

            # Check for missing data at this timestep
            found_fully_missing = False
            for var in data_vars:
                if timestep_ds[var].isnull().all():
                    found_fully_missing = True
                    break
            if found_fully_missing:
                discarded_count += 1
                continue

            # Handle longitude range conversion
            timestep_ds["longitude"] = (timestep_ds["longitude"] + 180) % 360 - 180

            # Create new dataset with data variables and coordinates
            new_ds = timestep_ds[data_vars + ["latitude", "longitude"]]

            # Select only the central 81 pixels along each axis
            lat_size = new_ds.sizes['ny']
            lon_size = new_ds.sizes['nx']
            lat_mid = lat_size // 2
            lon_mid = lon_size // 2
            new_ds = new_ds.isel(
                ny=slice(lat_mid - 40, lat_mid + 41),
                nx=slice(lon_mid - 40, lon_mid + 41)
            )

            dest_data_file = (
                dest_dir / f"{sid}_{pd.to_datetime(time).strftime('%Y%m%dT%H%M%S')}.nc"
            )
            new_ds.to_netcdf(dest_data_file)

            # Add metadata for this timestep using diagnostics data for the current timestep
            sample_metadata = {
                "source_name": source_name,
                "sid": sid,
                "time": time,
                "season": season,
                "storm_latitude": storm_meta["storm_latitude"][time_idx].item(),
                "storm_longitude": storm_meta["storm_longitude"][time_idx].item(),
                "intensity": storm_meta["intensity"][time_idx].item(),
                "data_path": str(dest_data_file),
            }

            # Fix longitude range for storm coordinates
            if sample_metadata["storm_longitude"] > 180:
                sample_metadata["storm_longitude"] -= 360

            sample_metadata_rows.append(sample_metadata)
    # Convert to DataFrame
    sample_metadata_df = pd.DataFrame(sample_metadata_rows)
    return sample_metadata_df, discarded_count


def process_era5(source_files, dest_dir, verbose=False, num_workers=0):
    """Process ERA5 source files into the common format."""
    # Create the destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    samples_metadata_path = dest_dir / "samples_metadata.json"
    # Remove the samples metadata file if it already exists
    if samples_metadata_path.exists():
        samples_metadata_path.unlink()

    source_name = "tc_primed_era5"
    data_vars = ["sst", "pressure_msl", "u_wind_10m", "v_wind_10m"]
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    context_vars = []

    # Write the source metadata file
    source_metadata = {
        "source_name": source_name,
        "source_type": "era5",
        "dim": 2,
        "data_vars": data_vars,
        "context_vars": context_vars,
        "storm_metadata_vars": storm_vars,
        "is_on_regular_grid": True,
    }
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)

    if num_workers < 1:
        # Process files sequentially
        results = []
        for file in tqdm(source_files, desc="Processing ERA5") if verbose else source_files:
            result = process_file(file, dest_dir, data_vars, source_name)
            results.append(result)
    else:
        # Process files in parallel using ProcessPoolExecutor
        process_file_partial = partial(process_file, dest_dir=dest_dir, data_vars=data_vars, source_name=source_name)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(executor.map(process_file_partial, source_files), total=len(source_files), desc="Processing ERA5")
                if verbose else executor.map(process_file_partial, source_files)
            )

    # Collect DataFrames and total discarded count
    metadata_dfs = [df for df, _ in results]
    total_discarded = sum(discarded for _, discarded in results)

    # Concatenate all metadata DataFrames
    combined_metadata_df = pd.concat(metadata_dfs, ignore_index=True)

    # Write the combined metadata DataFrame to JSON
    combined_metadata_df.to_json(
        samples_metadata_path,
        orient="records",
        lines=True,
        default_handler=str,
    )

    if total_discarded > 0:
        print(
            f"ERA5: Discarded {total_discarded} samples due to missing data "
            f"({total_discarded / len(source_files):.2%})"
        )


def process_storm_metadata(source_files, dest_dir, verbose=False):
    """Process storm metadata source files into the common format."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    samples_metadata_path = dest_dir / "samples_metadata.json"
    if samples_metadata_path.exists():
        samples_metadata_path.unlink()

    source_name = "tc_primed_storm_metadata"
    data_vars = ["intensity", "central_min_pressure", "storm_heading", "storm_speed"]
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    context_vars = []

    # Write the source metadata file
    source_metadata = {
        "source_name": source_name,
        "source_type": "storm_metadata",
        "dim": 0,
        "data_vars": data_vars,
        "context_vars": context_vars,
        "storm_metadata_vars": storm_vars,
        "is_on_regular_grid": True,
    }
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)

    iterator = tqdm(source_files, desc="Processing Storm Metadata") if verbose else source_files

    for file in iterator:
        with Dataset(file) as raw_sample:
            storm_meta = raw_sample["storm_metadata"]
            times = raw_sample["rectilinear"]["time"][:]

            # Get storm identification info
            season = storm_meta["season"][0].item()
            basin = storm_meta["basin"][0]
            storm_number = storm_meta["cyclone_number"][-1].item()
            sid = f"{season}{basin}{storm_number}"

            # Process each timestep
            for time_idx, time in enumerate(times):
                # Fix longitude range
                storm_lon = storm_meta["storm_longitude"][time_idx].item()
                storm_lon = storm_lon - 360 if storm_lon > 180 else storm_lon
                storm_lat = storm_meta["storm_latitude"][time_idx].item()

                # Create new dataset without dimensions or coordinates
                new_ds = xr.Dataset(
                    data_vars={
                        "intensity": storm_meta["intensity"][time_idx].item(),
                        "central_min_pressure": storm_meta["central_min_pressure"][time_idx].item(),
                        "storm_heading": storm_meta["storm_heading"][time_idx].item(),
                        "storm_speed": storm_meta["storm_speed"][time_idx].item(),
                        "time": pd.to_datetime(time),
                        "latitude": storm_lat,
                        "longitude": storm_lon,
                    }
                )

                dest_data_file = (
                    dest_dir / f"{sid}_{pd.to_datetime(time).strftime('%Y%m%dT%H%M%S')}.nc"
                )
                new_ds.to_netcdf(dest_data_file)

                # Add metadata for this timestep
                sample_metadata = pd.DataFrame(
                    {
                        "source_name": source_name,
                        "sid": sid,
                        "time": time,
                        "season": season,
                        "storm_latitude": storm_lat,
                        "storm_longitude": storm_lon,
                        "intensity": storm_meta["intensity"][time_idx].item(),
                        "data_path": dest_data_file,
                    },
                    index=[0],
                )

                sample_metadata.to_json(
                    samples_metadata_path,
                    orient="records",
                    lines=True,
                    mode="a",
                    default_handler=str,
                )


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"

    # Get ERA5 files
    sources, source_files, _ = list_tc_primed_sources(tc_primed_path, source_type="environmental")
    era5_files = source_files.get("era5", [])

    num_workers = cfg.get("num_workers", 0)

    if era5_files:
        # Process ERA5 source
        era5_dest_dir = dest_path / "tc_primed_era5"
        process_era5(era5_files, era5_dest_dir, verbose=True, num_workers=num_workers)

        # Process storm metadata source
        storm_meta_dest_dir = dest_path / "tc_primed_storm_metadata"
        process_storm_metadata(era5_files, storm_meta_dest_dir, verbose=True)


if __name__ == "__main__":
    main()
