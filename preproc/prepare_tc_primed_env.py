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


def process_era5(source_files, dest_dir, verbose=False):
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
    }
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)

    # Process the files sequentially
    iterator = tqdm(source_files, desc="Processing ERA5") if verbose else source_files
    discarded_count = 0

    for file in iterator:
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

                dest_data_file = (
                    dest_dir / f"{sid}_{pd.to_datetime(time).strftime('%Y%m%dT%H%M%S')}.nc"
                )
                new_ds.to_netcdf(dest_data_file)

                # Add metadata for this timestep using diagnostics data for the current timestep
                sample_metadata = pd.DataFrame(
                    {
                        "source_name": source_name,
                        "sid": sid,
                        "time": time,
                        "season": season,
                        "storm_latitude": storm_meta["storm_latitude"][time_idx].item(),
                        "storm_longitude": storm_meta["storm_longitude"][time_idx].item(),
                        "intensity": storm_meta["intensity"][time_idx].item(),
                        "data_path": dest_data_file,
                    },
                    index=[0],
                )

                # Fix longitude range
                sample_metadata["storm_longitude"] = sample_metadata["storm_longitude"].apply(
                    lambda x: x - 360 if x > 180 else x
                )

                # Append metadata
                sample_metadata.to_json(
                    samples_metadata_path,
                    orient="records",
                    lines=True,
                    mode="a",
                    default_handler=str,
                )

    if discarded_count > 0:
        print(
            f"ERA5: Discarded {discarded_count} samples due to missing data "
            f"({discarded_count/len(source_files):.2%})"
        )


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"

    # Get ERA5 files only
    sources, source_files, _ = list_tc_primed_sources(tc_primed_path, source_type="environmental")
    era5_files = source_files.get("era5", [])

    if era5_files:
        dest_dir = dest_path / "tc_primed_era5"
        process_era5(era5_files, dest_dir, verbose=True)


if __name__ == "__main__":
    main()
