"""Usage: python prepare_tc_primed.py (meant to be used with hydra).
Formats the TC-Primed dataset into a format that can be used by the following preprocessing
scripts that are common to all sources. That destination format is a directory containing
a subdir for each source, as:
    - tc_primed_{source}/
        - source_metadata.json
            * source_name: tc_primed_{source}
            * source_type: {radar, pmw, infrared}
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
    - radar_{sensor}_{swath} --> The radar-radiometer data from the sensor and swath.
    - pmw_{sensor}_{swath} --> Means that different swaths from the same sensor are treated as
        different sources, although taken at nearly the same time.
    - infrared
"""

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from netCDF4 import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from xarray.backends import NetCDF4DataStore

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


def initialize_source_metadata(ds, source, dest_dir):
    """Initializes source metadata by processing the first file."""
    data_vars = None
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]

    # For passive microwave data
    if source.startswith("pmw_"):
        data_vars = [var for var in ds.data_vars if var.startswith("TB")]
        charac_vars = ["frequency"]
    # For radar-radiometer data
    elif source.startswith("radar_"):
        data_vars = ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma"]
        charac_vars = []
    # For infrared data
    elif source == "infrared":
        data_vars = ["IRWIN"]
        charac_vars = []

    # Add IFOV values as characteristic variables for satellite data
    ifov_vars = [
        "IFOV_nadir_along_track",
        "IFOV_nadir_across_track",
        "IFOV_edge_along_track",
        "IFOV_edge_across_track",
    ]
    charac_vars += ifov_vars

    # Create the destination directory and initialize samples metadata file
    dest_dir.mkdir(parents=True, exist_ok=True)
    samples_metadata_path = dest_dir / "samples_metadata.json"
    if samples_metadata_path.exists():
        samples_metadata_path.unlink()

    # Write the source metadata file
    source_metadata = {
        "source_name": "tc_primed_" + source,
        "source_type": source.split("_")[0] if source != "infrared" else "infrared",
        "dim": 2,
        "data_vars": data_vars,
        "charac_vars": charac_vars,
        "storm_metadata_vars": storm_vars,
        "is_on_regular_grid": source == "infrared",
    }
    with open(dest_dir / "source_metadata.json", "w") as f:
        json.dump(source_metadata, f)

    return data_vars, charac_vars, ifov_vars, samples_metadata_path


def initialize_all_sources_metadata(sources, source_files, source_groups, dest_path, ifovs):
    """Initialize metadata for all sources before processing."""
    metadata_dict = {}
    for source in sources:
        if source != "infrared":
            source_parts = source.split("_")
            instrument = "_".join(source_parts[1:-1])
            swath = source_parts[-1]
            if instrument not in ifovs or swath not in ifovs[instrument]:
                continue

        dest_dir = dest_path / f"tc_primed_{source}"

        # Process first file to get metadata
        with Dataset(source_files[source][0]) as raw_sample:
            ds = raw_sample
            for group in source_groups[source]:
                if group in ds.groups:
                    ds = ds[group]
            ds = xr.open_dataset(NetCDF4DataStore(ds), decode_times=False)
            ds["time"] = pd.to_datetime(ds["time"], origin="unix", unit="s")

            if source != "infrared":
                ds = preprocess_satellite_source(ds)

            metadata = initialize_source_metadata(ds, source, dest_dir)
            metadata_dict[source] = metadata

    return metadata_dict


def process_source(source, source_files, source_groups, dest_dir, ifovs, metadata, verbose=False):
    """Processes a single source."""
    if source != "infrared":
        # Split source into type and instrument (e.g. "pmw_AMSR2_GCOMW1_S1" -> ("pmw", "AMSR2_GCOMW1", "S1"))
        source_parts = source.split("_")
        source_type = source_parts[0]
        instrument = "_".join(
            source_parts[1:-1]
        )  # Handle instruments with underscores like "AMSR2_GCOMW1"
        swath = source_parts[-1]

        # Skip sources not in IFOV config
        if instrument not in ifovs:
            return
        # Skip if swath not in instrument's config
        if swath not in ifovs[instrument]:
            return
    else:
        source_type = "infrared"
        instrument = None
        swath = None

    source_name = "tc_primed_" + source
    data_vars, context_vars, ifov_vars, samples_metadata_path = metadata[source]

    # Process the files sequentially
    iterator = tqdm(source_files, desc=f"Processing {source}") if verbose else source_files
    discarded_count = 0

    for file in iterator:
        with Dataset(file) as raw_sample:
            # For radar data, check the availability flag
            if source.startswith("radar_"):
                if raw_sample["radar_radiometer"]["availability_flag"][0].item() == 0:
                    continue
            # For infrared data, check the availability flag. 0 means the infrared
            # is unavailable, 2 means it's from HURSAT. We'll only use it when it's
            # 1, which means it's from TC IRAR.
            if source == "infrared":
                if raw_sample["infrared"]["infrared_availability_flag"][0].item() in [0, 2]:
                    continue
            # Access the group containing the data
            ds = raw_sample
            for group in source_groups:
                if group in ds.groups:
                    ds = ds[group]
            # Open the dataset as an xarray Dataset
            ds = xr.open_dataset(NetCDF4DataStore(ds), decode_times=False)
            ds["time"] = pd.to_datetime(ds["time"], origin="unix", unit="s")
            if source != "infrared":
                ds = preprocess_satellite_source(ds)

            if ds is None:  # If the data is not available, skip this sample
                continue

            # Check if any of the data variables are fully missing. If so, discard the
            # sample. Also keep count of how many samples were discarded, to print at the end.
            found_fully_missing = False
            for var in data_vars:
                if ds[var].isnull().all():
                    found_fully_missing = True
                    break
            if found_fully_missing:
                discarded_count += 1
                continue

            # Load the storm metadata variables
            if source == "infrared":
                # For the infrared source, we use the storm metadata from the
                # "infrared" group directly.
                raw_storm_meta = ds
            else:
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
                    "source_type": source_type,
                    "sid": sid,
                    "time": time,
                    "season": season,
                    "storm_latitude": raw_storm_meta["storm_latitude"][0].item(),
                    "storm_longitude": raw_storm_meta["storm_longitude"][0].item(),
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
            if source == "infrared":
                swath_ifovs = ifovs["infrared"]
            else:
                swath_ifovs = ifovs[instrument][swath]
            if isinstance(swath_ifovs, list):
                # Case where the swath has a single list of IFOVs for all channels
                for i, cvar in enumerate(ifov_vars):
                    sample_metadata.at[0, cvar] = {var: float(swath_ifovs[i]) for var in data_vars}
            else:
                # Case where the swath has channel-specific IFOVs
                for i, cvar in enumerate(ifov_vars):
                    sample_metadata.at[0, cvar] = {
                        var: float(swath_ifovs[var][i]) for var in data_vars if var in swath_ifovs
                    }

            # Append the row to the samples metadata file
            sample_metadata.to_json(
                samples_metadata_path, orient="records", lines=True, mode="a", default_handler=str
            )

    return discarded_count


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
    # Optionally: if process_only is a str, only process sources that contain that string.
    process_only = cfg.get("process_only", None)

    # Retrieve the list of files from the TC-Primed dataset, excluding environmental sources
    sources, source_files, source_groups = list_tc_primed_sources(
        tc_primed_path, source_type="satellite"
    )
    if process_only is not None:
        sources = [source for source in sources if process_only in source]
        source_files = {source: source_files[source] for source in sources}
        source_groups = {source: source_groups[source] for source in sources}

    # Initialize metadata for all sources first
    metadata_dict = initialize_all_sources_metadata(
        sources, source_files, source_groups, dest_path, ifovs
    )

    # Process each source sequentially, but parallelize file processing within each source
    if "num_workers" in cfg and cfg["num_workers"] > 1:
        for source in sources:
            if source not in metadata_dict:  # Skip sources that were filtered out
                continue
            dest_dir = dest_path / f"tc_primed_{source}"

            # Split files into chunks
            files = source_files[source]
            file_chunks = np.array_split(files, cfg["num_workers"])

            print(f"Processing {source} with {len(file_chunks)} workers")

            with ProcessPoolExecutor(max_workers=cfg["num_workers"]) as executor:
                futures = [
                    executor.submit(
                        process_source,
                        source,
                        chunk,
                        source_groups[source],
                        dest_dir,
                        ifovs,
                        metadata_dict,
                        False,
                    )
                    for chunk in file_chunks
                ]

                # Sum up discarded counts from all workers
                total_discarded = sum(
                    future.result()
                    for future in tqdm(futures, total=len(futures), desc=f"Processing {source}")
                )

                if total_discarded > 0:
                    discarded_fraction = total_discarded / len(files)
                    print(
                        f"tc_primed_{source}: Discarded {total_discarded} samples due "
                        f"to missing data ({discarded_fraction:.2%})"
                    )

                # Remove directory if all files were discarded
                if total_discarded == len(files):
                    print(f"Removing {dest_dir} as all files were discarded")
                    shutil.rmtree(dest_dir)
    else:
        for source in sources:
            if source not in metadata_dict:  # Skip sources that were filtered out
                continue
            dest_dir = dest_path / f"tc_primed_{source}"
            discarded_count = process_source(
                source,
                source_files[source],
                source_groups[source],
                dest_dir,
                ifovs,
                metadata_dict,
                verbose=True,
            )

            if discarded_count > 0:
                discarded_fraction = discarded_count / len(source_files[source])
                print(
                    f"tc_primed_{source}: Discarded {discarded_count} samples due "
                    f"to missing data ({discarded_fraction:.2%})"
                )

            # Remove directory if all files were discarded
            if discarded_count == len(source_files[source]):
                print(f"Removing {dest_dir} as all files were discarded")
                shutil.rmtree(dest_dir)


if __name__ == "__main__":
    main()
