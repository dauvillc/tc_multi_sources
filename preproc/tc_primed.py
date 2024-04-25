"""
Clément Dauvilliers - 23/04/2024.
Preprocesses the TC-Primed dataset by normalizing the data, treating missing values,
and formatting the result into the sources tree.
"""

import yaml
import xarray as xr
import netCDF4 as nc
import numpy as np
from dask.diagnostics import ProgressBar
from xarray.core.dtypes import NA
from pathlib import Path
from tqdm import tqdm


def pad_dataset(ds, max_size):
    """
    Pads the dataset ds to the size max_size, by adding missing values
    at the bottom and right of the dataset.
    """
    # Retrieve the size of the dataset
    sizes = ds.sizes
    size = (sizes["scan"], sizes["pixel"])
    # Compute the padding values
    pad_scan = max_size[0] - size[0]
    pad_pixel = max_size[1] - size[1]
    # Set all -9999.9 values to NaN
    ds = ds.where(ds != -9999.9)
    # Pad the dataset with NaN values, which is the missing value in the dataset
    # for Microwave brightness temperatures
    # Note: this ALSO pads the x, y, latitude and longitude variables !
    return ds.pad(
        {"scan": (0, pad_scan), "pixel": (0, pad_pixel)}, mode="constant", constant_values=NA
    )


def main():
    # Load the paths configuration file
    with open("paths.yml", "r") as file:
        paths_cfg = yaml.safe_load(file)
    tc_primed_path = Path(paths_cfg["raw_datasets"]) / "tc_primed"
    dest_path = Path(paths_cfg["sources"]) / "tc_primed"
    # Load the sources configuration file
    with open("sources.yml", "r") as file:
        sources_cfg = yaml.safe_load(file)["tc_primed"]
        sen_sat_pairs = sources_cfg["sensor_satellite_pairs"]
        # The sen/sat pairs are dicts of the form
        # SENSOR_SATELLITE: {swath1: [bands], swath2: [bands], ...}
        # If sen_sat_pairs is None, process all sensor/satellite pairs
    # The sources configuration contains the list of pairs (sensor, satellite) to include.
    # The raw dataset has the structure tc_primed/{year}/{basin}/{number}/{filename}.nc
    # Retrieve all filenames for all years
    all_files = []
    for year in tc_primed_path.iterdir():
        for basin in year.iterdir():
            for number in basin.iterdir():
                all_files.extend(number.glob("*.nc"))
    # The filenames are formatted as:
    # - TCPRIMED_VERSION_BASINNUMBERYEAR_SENSOR_SATELLITE_IMGNUMBER_YYYYMMDDHHmmSS.nc
    #   for the overpass files;
    # - TCPRIMED_VERSION_BASINNUMBERYEAR_era5_START_END.nc for the environmental files.
    # Isolate the overpass files:
    overpass_files = [file for file in all_files if "era5" not in file.stem]
    # Deduce the list of strings {sensor}_{satellite} from the filenames
    available_sen_sat = set()
    for file in overpass_files:
        available_sen_sat.add("_".join(file.stem.split("_")[3:5]))
    # If the list of pairs is not specified, process all available pairs
    # and for each pair, process all swaths and bands
    if sen_sat_pairs is None:
        sen_sat_pairs = {sensat: None for sensat in available_sen_sat}
    # Otherwise, check that the pairs in the configuration file are in the dataset
    else:
        for pair in sen_sat_pairs.keys():
            if pair not in available_sen_sat:
                raise ValueError(f"Sensor/Satellite pair {pair} not found in the dataset.")
    # For each sensor/satellite pair, preprocess the data
    for sensat in sen_sat_pairs.keys():
        print(f"Processing sensor/satellite pair {sensat}")
        # Retrieve the list of files whose stem contains the sensor/satellite pair
        files = [file for file in overpass_files if sensat in file.stem]
        # If the list of swaths is not specified, we need to open a file with netCDF4 to
        # retrieve the list of swaths
        if sen_sat_pairs[sensat] is None:
            with nc.Dataset(files[0], "r") as ds:
                swaths = [swath for swath in ds["passive_microwave"].groups.keys()]
        # Otherwise, use the list of swaths in the configuration file
        else:
            swaths = list(sen_sat_pairs[sensat].keys())
        # For each swath, preprocess the data
        for swath in swaths:
            print(f"Processing swath {swath}")
            # Create the destination directory dest_path/under microwave/SENSOR_SATELLITE/swath/
            dest_swath_dir = dest_path / "microwave" / sensat / swath
            dest_swath_dir.mkdir(parents=True, exist_ok=True)
            # Although a swath always contains the same bands, the size of the images is not
            # identical across the samples, due to missing values that have been removed.
            # Therefore, we can't stack the images into a single xarray dataset just yet.
            # To do so, we'll pad all images from the same swath to the same size. That size
            # is the maximum size of all images from the swath.
            # - Retrieve the maximum size of the images from the swath
            max_size = [0, 0]
            print("Retrieving maximum size of images")
            for file in tqdm(files):
                with nc.Dataset(file, "r") as ds:
                    dims = ds["passive_microwave"][swath].dimensions
                    size = (dims["scan"].size, dims["pixel"].size)
                    max_size[0] = max(max_size[0], size[0])
                    max_size[1] = max(max_size[1], size[1])
            # - Load all images after padding them via the preprocessing function
            print("Preparing operations")

            def _preprocess(ds):
                """Pads the dataset ds to the size max_size.
                Note: also pads the x, y, latitude and longitude variables."""
                # Discard the 'ScanTime' and 'angle_bins' variables
                ds = ds.drop_vars(["ScanTime", "angle_bins"])
                # Set the variable 'x'
                return pad_dataset(ds, max_size)

            dataset = xr.open_mfdataset(
                files,
                concat_dim="sample",
                combine="nested",
                group=f"passive_microwave/{swath}",
                preprocess=_preprocess,
                parallel=True,
            )
            # Normalization:
            # - Compute the mean and standard deviation of each band, without considering missing values
            # - Set the missing values to 0.
            # - Normalize the data by subtracting the mean and dividing by the standard deviation
            mean = dataset.mean(dim=["sample", "scan", "pixel"], skipna=True)
            std = dataset.std(dim=["sample", "scan", "pixel"], skipna=True)
            # To reinitialize the chunking, xarray advise to write the mean/std to a file and reload them.
            print("Computing normalization parameters")
            with ProgressBar():
                mean.to_netcdf(dest_swath_dir / "normalization_mean.nc")
                std.to_netcdf(dest_swath_dir / "normalization_std.nc")
            # Reload the mean and standard deviation
            mean = xr.open_dataset(dest_swath_dir / "normalization_mean.nc")
            std = xr.open_dataset(dest_swath_dir / "normalization_std.nc")
            # Set the missing values to 0
            dataset = dataset.fillna(0)
            # Normalize the data
            dataset = (dataset - mean) / std
            # Now, load the overpass_metadata group of the files, which notably include
            # the overpass time, basin, year and storm number.
            meta = xr.concat(
                [xr.open_dataset(file, group="overpass_metadata") for file in files],
                dim="time",
            ).rename_dims({"time": "sample"}).reset_index("time").reset_coords()
            meta = meta.set_coords(["season", "basin", "cyclone_number", "time"])
            # Add the time, basin, cyclone_number, season, instrument_name, platform_name,
            # coverage_fraction and ERA5_time variables to the dataset
            meta_vars = [
                "time",
                "basin",
                "cyclone_number",
                "season",
                "instrument_name",
                "platform_name",
                "coverage_fraction",
                "ERA5_time",
            ]
            dataset = dataset.assign({var: meta[var] for var in meta_vars})
            # Note: the previous line also sets meta's coordinates as coordinates for dataset.

            # The results will be stored under
            # dest_path/microwave/SENSOR_SATELLITE/swath/YEAR/BASIN/NUMBER.nc
            # Where NUMBER.nc is the concatenation of all images from the same storm,
            # along the time dimension.
            # - Isolate the unique seasons
            seasons = np.unique(dataset["season"].data)
            for season in seasons:
                print("Processing season ", season)
                # - Isolate the unique basins
                season_ds = dataset.where(dataset["season"] == season, drop=True)
                basins = np.unique(season_ds["basin"].data)
                for basin in basins:
                    print("Processing basin ", basin)
                    # Create the destination directory dest_path/microwave/SENSOR_SATELLITE/swath/YEAR/BASIN/
                    (dest_swath_dir / f"{season}" / f"{basin}").mkdir(parents=True, exist_ok=True)
                    # - Isolate the unique cyclone numbers
                    basin_ds = season_ds.where(season_ds["basin"] == basin, drop=True)
                    cyclone_numbers = np.unique(basin_ds["cyclone_number"].data)
                    for cyclone_number in tqdm(cyclone_numbers):
                        cyc_ds = basin_ds.where(basin_ds["cyclone_number"] == cyclone_number, drop=True)
                        if cyc_ds["sample"].size == 0:
                            continue
                        # Sort the dataset by the "time" coordinate
                        cyc_ds = cyc_ds.sortby("time")
                        # Write the dataset to the destination file
                        cyc_ds.to_netcdf(dest_swath_dir / f"{season}" / f"{basin}" / f"{cyclone_number}.nc")

            return 0


if __name__ == "__main__":
    main()
