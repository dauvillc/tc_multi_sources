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


def reverse_spatially(ds):
    """
    Reverses the dataset ds spatially, i.e. reverses the pixel and scan dimensions
    if the image was taken on the descending pass.
    """
    # The latitude variable in TC-PRIMEd is in degrees north, so it should be
    # decreasing from the top to the bottom of the image.
    if ds.latitude[0, 0] < ds.latitude[1, 0]:
        return ds.isel(scan=slice(None, None, -1), pixel=slice(None, None, -1))
    return ds


def main():
    # Load the paths configuration file
    with open("paths.yml", "r") as file:
        paths_cfg = yaml.safe_load(file)
    tc_primed_path = Path(paths_cfg["raw_datasets"]) / "tc_primed"
    dest_path = Path(paths_cfg["sources"]) / "tc_primed"
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
    sen_sat_pairs = set()
    for file in overpass_files:
        sen_sat_pairs.add("_".join(file.stem.split("_")[3:5]))
    # For each sensor/satellite pair, preprocess the data
    for sensat in sen_sat_pairs:
        print(f"Processing sensor/satellite pair {sensat}")
        # Retrieve the list of files whose stem contains the sensor/satellite pair
        files = [file for file in overpass_files if sensat in file.stem]
        # We need to open a file with netCDF4 to retrieve the list of swaths
        with nc.Dataset(files[0], "r") as ds:
            swaths = [swath for swath in ds["passive_microwave"].groups.keys()]
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
                # Discard the 'ScanTime' and 'angle_bins' variables
                ds = ds.drop_vars(["ScanTime", "angle_bins"])
                # Reverse the dataset spatially if the image was taken on the descending pass
                ds = reverse_spatially(ds)
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
            # Normalize the data
            dataset = (dataset - mean) / std
            # Set the missing values to 0
            dataset = dataset.fillna(0)
            # Now, load the overpass_metadata group of the files, which notably include
            # the overpass time, basin, year and storm number.
            meta = (
                xr.concat(
                    [xr.open_dataset(file, group="overpass_metadata") for file in files],
                    dim="time",
                )
                .rename_dims({"time": "sample"})
                .reset_index("time")
                .reset_coords()
            )
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
                "ERA5_time",
                "coverage_fraction",
            ]
            dataset = dataset.assign({var: meta[var] for var in meta_vars})
            # Note: the previous line also sets meta's coordinates as coordinates for dataset.

            # Transform the "intstrument_name" and "platform_name" variables into a single
            # "sensor_satellite" variable
            dataset["sensor_satellite"] = np.repeat(
                dataset["instrument_name"][0].item() + "_" + dataset["platform_name"][0].item(),
                dataset["time"].size,
            )
            dataset = dataset.drop_vars(["instrument_name", "platform_name"])

            # Compute the "dist_to_storm_center" variable from x and y
            # where x is the east-west distance from the storm center. Note: this results in an about
            # 0.5% error compared to using the ellipsoidal distance
            # (see https://geopy.readthedocs.io/en/stable/#module-geopy.distance)
            dataset["dist_to_storm_center"] = (dataset['x'] ** 2 + dataset['y'] ** 2) ** 0.5

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
                        cyc_ds = basin_ds.where(
                            basin_ds["cyclone_number"] == cyclone_number, drop=True
                        )
                        if cyc_ds["sample"].size == 0:
                            continue
                        # Sort the dataset by the "time" coordinate
                        cyc_ds = cyc_ds.sortby("time")
                        # Write the dataset to the destination file
                        cyc_ds.to_netcdf(
                            dest_swath_dir / f"{season}" / f"{basin}" / f"{cyclone_number}.nc"
                        )


if __name__ == "__main__":
    main()
