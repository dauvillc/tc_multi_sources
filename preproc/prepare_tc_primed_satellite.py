"""Usage: python prepare_tc_primed.py (meant to be used with hydra).
Formats the TC-Primed dataset into a format that can be used by the following preprocessing
scripts that are common to all sources. That destination format is a directory containing
a subdir for each source, as:
    - tc_primed_{source}/
        - source_metadata.json
            * source_name: tc_primed_{source}
            * source_type: {radar, pmw}
            * dim: 2
            * data_vars: list of data variables
            * context_vars: list of context variables that describe each variable
                in data_vars.
            * storm_metadata_vars: list of variables that describe the storm at each sample.
                The included variables are: "storm_latitude", "storm_longitude", "intensity".
        - samples_metadata.json
            * data_path: path to the preprocessed data file
            * source_name: tc_primed_{source}
            * sid: storm ID
            * time: sample time
            * season: season of the storm
            * storm_latitude: latitude of the storm
            * storm_longitude: longitude of the storm
            * intensity: intensity of the storm
        - {sid}_{time}.nc
            * The preprocessed data file, containing a single observation from the source.
                The spatial dimensions are NOT in general
                the latitude and longitude, as the satellite files are still in satellite
                geometry.
The included sources are (for now):
    - radar_radiometer
    - pmw_{sensor}_{swath} --> Means that different swaths from the same sensor are treated as
        different sources, although taken at nearly the same time.
"""

import hydra
import yaml
import pandas as pd
import xarray as xr
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from netCDF4 import Dataset
from xarray.backends import NetCDF4DataStore
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from preproc.utils import list_tc_primed_sources


def parse_frequency(pm_var_name):
    """Parses the frequency of a passive microwave variable from its name."""
    # The var is named TB_<frequency><polarization>
    # or TB_<A/B><frequency><polarization>
    # or TB_<frequency>_<sideband><polarization> (i.e. TB_183.31_1.0QH)
    freq = pm_var_name.split("_")[1]
    for char in ["A", "B", "H", "V", "Q"]:
        freq = freq.replace(char, "")
    return float(freq)


def preprocess_satellite_source(ds):
    """Preprocesses satellite source data by dropping unused variables and standardizing dimensions."""
    ds = ds.drop_vars(["ScanTime", "angle_bins", "x", "y"])
    # For passive microwave data the dimensions are named "scan" and "pixel",
    # while for radar-radiometer they're named "scan" and "beam". We'll rename
    # the "beam" dimension to "pixel" for consistency.
    if "beam" in ds.dims:
        ds = ds.rename({"beam": "pixel"})
    return ds


def process_source(source, source_files, source_groups, dest_dir, ifovs, verbose=False):
    """Processes a single source."""
    # Split source into type and instrument (e.g. "pmw_AMSR2_GCOMW1_S1" -> ("pmw", "AMSR2_GCOMW1", "S1"))
    source_parts = source.split("_")
    source_type = source_parts[0]
    instrument = "_".join(source_parts[1:-1])  # Handle instruments with underscores like "AMSR2_GCOMW1"
    swath = source_parts[-1]
    
    # Skip sources not in IFOV config
    if instrument not in ifovs:
        return
    # Skip if swath not in instrument's config
    if swath not in ifovs[instrument]:
        return

    # Create the destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    samples_metadata_path = dest_dir / "samples_metadata.json"
    # Remove the samples metadata file if it already exists
    if samples_metadata_path.exists():
        samples_metadata_path.unlink()

    source_name = "tc_primed_" + source
    # List of data variables that will be kept in the preprocessed dataset
    data_vars = None
    # List of variables that give information about the storm at that sample
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]

    # Process the files sequentially
    iterator = tqdm(source_files, desc=f"Processing {source}") if verbose else source_files
    discarded_count = 0

    for file in iterator:
        with Dataset(file) as raw_sample:
            # Access the group containing the data
            ds = raw_sample
            for group in source_groups:
                ds = ds[group]
            # Open the dataset as an xarray Dataset
            ds = xr.open_dataset(NetCDF4DataStore(ds))
            ds = preprocess_satellite_source(ds)

            if ds is None:  # If the data is not available, skip this sample
                continue

            if data_vars is None:  # Only executed for the first file
                # For passive microwave data
                if source.startswith("pmw_"):
                    data_vars = [var for var in ds.data_vars if var.startswith("TB")]
                    context_vars = ["frequency"]
                # For radar-radiometer data
                elif source.startswith("radar_"):
                    data_vars = ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma"]
                    context_vars = []

                # Add IFOV values as context variables for satellite data
                ifov_vars = [
                    "IFOV_nadir_along_track",
                    "IFOV_nadir_across_track",
                    "IFOV_edge_along_track",
                    "IFOV_edge_across_track",
                ]
                context_vars += ifov_vars

                # Write the source metadata file
                source_metadata = {
                    "source_name": source_name,
                    "source_type": source_type,
                    "dim": 2,
                    "data_vars": data_vars,
                    "context_vars": context_vars,
                    "storm_metadata_vars": storm_vars,
                }
                with open(dest_dir / "source_metadata.json", "w") as f:
                    json.dump(source_metadata, f)

            # Check for missing data
            found_fully_missing = False
            for var in data_vars:
                if ds[var].isnull().all():
                    found_fully_missing = True
                    break
            if found_fully_missing:
                discarded_count += 1
                continue

            # Load the storm metadata variables
            raw_storm_meta = raw_sample["overpass_storm_metadata"]
            # Load the overpass metadata
            raw_overpass_meta = raw_sample["overpass_metadata"]
            season = raw_overpass_meta["season"][0].item()
            basin = raw_overpass_meta["basin"][0]
            storm_number = raw_overpass_meta["cyclone_number"][-1].item()
            sid = f"{season}{basin}{storm_number}"
            time = pd.to_datetime(raw_overpass_meta["time"][0], origin="unix", unit="s")

            # The longitudes in tc-primed as stored as [0, 360], but the libraries
            # we'll work with use [-180, 180].
            ds["longitude"] = (ds["longitude"] + 180) % 360 - 180

            # Create a new xr Dataset with the data variables, as well as the latitude
            # and longitude.
            new_ds = ds[data_vars + ["latitude", "longitude"]]

            dest_data_file = dest_dir / f"{sid}_{time.strftime('%Y%m%dT%H%M%S')}.nc"
            new_ds.to_netcdf(dest_data_file)

            # Add a row for that sample in the samples metadata file
            sample_metadata = pd.DataFrame(
                {
                    "source_name": source_name,
                    "sid": sid,
                    "time": time,
                    "season": season,
                    "storm_latitude": raw_storm_meta["storm_latitude"][0].item(),
                    "storm_longitude": raw_storm_meta["storm_longitude"][0].item(),
                    "intensity": raw_storm_meta["intensity"][0].item(),
                    "data_path": dest_data_file,
                },
                index=[0],
            )
            # Make sure the storm longitude is within the range [-180, 180]
            sample_metadata["storm_longitude"] = sample_metadata["storm_longitude"].apply(
                lambda x: x - 360 if x > 180 else x
            )
            # Initialize the context variables to None, which will allow us to
            # set the values of cells to lists with .at afterwards.
            for cvar in context_vars:
                sample_metadata[cvar] = ""
            # Set the frequency context variables
            if source.startswith("pmw_"):
                # doing sample_metadata['frequency'] = list would understand that we're trying
                # to create several rows, so we need to use 'at' instead.
                sample_metadata.at[0, "frequency"] = {
                    var: parse_frequency(var) for var in data_vars
                }

            # Handle IFOV context variables for satellite data
            swath_ifovs = ifovs[instrument][swath]
            if isinstance(swath_ifovs, list):
                # Case where the swath has a single list of IFOVs for all channels
                for i, cvar in enumerate(ifov_vars):
                    sample_metadata.at[0, cvar] = {
                        var: float(swath_ifovs[i]) for var in data_vars
                    }
            else:
                # Case where the swath has channel-specific IFOVs
                for i, cvar in enumerate(ifov_vars):
                    sample_metadata.at[0, cvar] = {
                        var: float(swath_ifovs[var][i]) 
                        for var in data_vars 
                        if var in swath_ifovs
                    }

            # Append the row to the samples metadata file
            sample_metadata.to_json(
                samples_metadata_path, orient="records", lines=True, mode="a", default_handler=str
            )

            new_ds.close()
            ds.close()

    if discarded_count > 0:
        discarded_fraction = discarded_count / len(source_files)
        print(
            f"{source_name}: Discarded {discarded_count} samples due \
to missing data ({discarded_fraction:.2%})"
        )
        # In some cases, the data might be fully missing, in which case
        # we'll remove the source directory.
        if discarded_fraction == 1:
            shutil.rmtree(dest_dir)


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # Path to the raw dataset and the file containing the IFOVs for each sensor-satellite pair
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    ifovs_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed_ifovs.yaml"
    with open(ifovs_path, "r") as f:
        ifovs = yaml.safe_load(f)
    # Path to where the prepared dataset will be stored (as netCDF files)
    dest_path = Path(cfg["paths"]["preprocessed_dataset"]) / "prepared"

    # Retrieve the list of files from the TC-Primed dataset, excluding environmental sources
    sources, source_files, source_groups = list_tc_primed_sources(
        tc_primed_path, 
        source_type="satellite"
    )

    # Process each source either sequentially or in parallel
    if "num_workers" in cfg:
        with ProcessPoolExecutor(max_workers=cfg["num_workers"]) as executor:
            futures = []
            for source in sources:
                dest_dir = dest_path / f"tc_primed_{source}"
                future = executor.submit(
                    process_source,
                    source,
                    source_files[source],
                    source_groups[source],
                    dest_dir,
                    ifovs,
                    verbose=False,
                )
                futures.append(future)
            for future in tqdm(futures, desc="Processing sources"):
                future.result()
    else:
        for source in sources:
            dest_dir = dest_path / f"tc_primed_{source}"
            process_source(
                source, source_files[source], source_groups[source], dest_dir, ifovs, verbose=True
            )


if __name__ == "__main__":
    main()
