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


def find_nc_group(swath):
    if swath in ["KuGMI", "KuTMI"]:
        return "radar_radiometer"
    return "passive_microwave"


def preprocess_source_files(ds, swath):
    group = find_nc_group(swath)
    # In some cases for radar-radiometer data, the radar swath is not available
    # because it doesn't include enough of the cyclone.
    if swath not in ds[group].groups:
        return None
    ds = xr.open_dataset(NetCDF4DataStore(ds[group][swath]))

    ds = ds.drop_vars(["ScanTime", "angle_bins", "x", "y"])
    # For passive microwave data the dimensions are named "scan" and "pixel",
    # while for radar-radiometer they're named "scan" and "beam". We'll rename
    # the "beam" dimension to "pixel" for consistency.
    if "beam" in ds.dims:
        ds = ds.rename({"beam": "pixel"})
    return ds


def process_swath(sensat, swath, sensat_files, dest_dir, ifovs, verbose=False):
    """Processes a single swath from a given sensor-satellite pair.
    Reads each file from the list of files, and creates the following files:
    - source_metadata.json with information about the source, with the following entry:
        source_name: str, type: str, dim: int, data_vars: list of str, context_vars: list of str,
        storm_metadata_vars: list of str
    - samples_metadata.json, which can be read as pandas DataFrame, with the following columns:
        source_name: str, sid: str, time: timestamp, season: int,
        context_var_0: list of float, ..., context_var_N: list of float,
            (for each var, one element for each variable in data_vars).
        storm_metadata_var_0: float, ..., storm_metadata_var_N: float,
        data_path: str
    - For each sample, a netCDF file with dims (latitude, longitude) and the following variables:
        - latitude, longitude: float32 (latitude, longitude)
        - data_var: float32 (data variable)
    """
    # Create the destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    samples_metadata_path = dest_dir / "samples_metadata.json"
    # Remove the samples metadata file if it already exists
    if samples_metadata_path.exists():
        samples_metadata_path.unlink()
    source_name = "tc_primed_" + sensat + "_" + swath
    # List of data variables that will be kept in the preprocessed dataset. To know
    # which variables are available, we'll read the first file.
    data_vars = None
    # List of variables that give information about the storm at that sample.
    storm_vars = ["storm_latitude", "storm_longitude", "intensity"]
    # Process the files sequentially
    iterator = tqdm(sensat_files, desc=f"Processing {sensat} {swath}") if verbose else sensat_files
    discarded_count = 0
    for file in iterator:
        with Dataset(file) as raw_sample:
            ds = preprocess_source_files(raw_sample, swath)
            if ds is None:  # If the swath is not available, we'll skip this sample
                continue

            if data_vars is None:  # Only executed for the first file
                group = find_nc_group(swath)
                # If the group is passive microwave, we'll keep all the variables that
                # begin with 'TB' (brightness temperature), and add the observing frequency
                # as a context variable.
                # The context variables will stored as a dict {var_name: [values]}. For context
                # variables that are the same for all data variables, we'll repeat the same value.
                if group == "passive_microwave":
                    data_vars = [var for var in ds.data_vars if var.startswith("TB")]
                    context_vars = ["frequency"]
                # If the group is radar-radiometer, we'll keep the near-surface rain rate
                # and its uncertainty. There are no context variables.
                elif group == "radar_radiometer":
                    data_vars = ["nearSurfPrecipTotRate", "nearSurfPrecipTotRateSigma"]
                    context_vars = []

                # We'll add the four IFOV values as context variables
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
                    "source_type": group,
                    "dim": 2,
                    "data_vars": data_vars,
                    "context_vars": context_vars,
                    # frequency of each variable in data_vars
                    "storm_metadata_vars": storm_vars,
                }
                with open(dest_dir / "source_metadata.json", "w") as f:
                    json.dump(source_metadata, f)

            # In some very rare cases, one of the swath will be only missing data.
            # This can happen when part of the data was unavailable but the largest
            # swath of an overpass still included more than 50% of the cyclone area.
            # We'll discard those cases so that the principle "all swaths are available"
            # is alway respected. Those are an absolute minority of cases.
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
            if group == "passive_microwave":
                # doing sample_metadata['frequency'] = list would understand that we're trying
                # to create several rows, so we need to use 'at' instead.
                sample_metadata.at[0, "frequency"] = {
                    var: parse_frequency(var) for var in data_vars
                }

            # For each data variable, add the IFOV values as context variables.
            ifov = ifovs[sensat][swath]
            if isinstance(ifov, list):
                # If ifov is a list, then all data variables have the same IFOV values.
                # In this case, we'll just repeat the same values for each data variable.
                for i, cvar in enumerate(ifov_vars):
                    sample_metadata.at[0, cvar] = {var: float(ifov[i]) for var in data_vars}
            else:
                # If ifov is a dict, then each data variable has its own IFOV values.
                for i, cvar in enumerate(ifov_vars):
                    sample_metadata.at[0, cvar] = {var: float(ifov[var][i]) for var in data_vars}

            # Append the row to the samples metadata file
            sample_metadata.to_json(
                samples_metadata_path, orient="records", lines=True, mode="a", default_handler=str
            )

            new_ds.close()
            ds.close()

    if discarded_count > 0:
        discarded_fraction = discarded_count / len(sensat_files)
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

    # Retrieve the list of files from the TC-Primed dataset, excluding the validation and test sets
    sen_sat_pairs, sen_sat_files, sen_sat_swaths = list_tc_primed_sources(tc_primed_path)

    # Process each sensor-satellite pair either sequentially or in parallel
    if "num_workers" in cfg:
        with ProcessPoolExecutor(max_workers=cfg["num_workers"]) as executor:
            futures = []
            for sensat in sen_sat_pairs:
                for swath in sen_sat_swaths[sensat]:
                    dest_dir = dest_path / f"tc_primed_{sensat}_{swath}"
                    future = executor.submit(
                        process_swath,
                        sensat,
                        swath,
                        sen_sat_files[sensat],
                        dest_dir,
                        ifovs,
                        verbose=False,
                    )
                    futures.append(future)
            for future in tqdm(futures, desc="Processing sensor-satellite pairs"):
                future.result()
    else:
        for sensat in sen_sat_pairs:
            for swath in sen_sat_swaths[sensat]:
                # Create the destination directory for this sensor-satellite pair
                dest_dir = dest_path / f"tc_primed_{sensat}_{swath}"
                process_swath(sensat, swath, sen_sat_files[sensat], dest_dir, ifovs, verbose=True)


if __name__ == "__main__":
    main()
