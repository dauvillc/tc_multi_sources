"""
Clément Dauvilliers - 23/04/2024.
Preprocesses the TC-Primed dataset by normalizing the data, treating missing values,
and formatting the result into the sources tree.
"""

import hydra
import xarray as xr
import netCDF4 as nc
import numpy as np
from global_land_mask import globe
from dask.diagnostics import ProgressBar
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from xarray.core.dtypes import NA
from pathlib import Path
from preproc.utils import list_tc_primed_sources, list_tc_primed_storm_files


# Order of the metadata columns in the CSV file
_METADATA_COL_ORDER_ = [
    "sid",
    "time",
    "season",
    "basin",
    "cyclone_number",
    "source_name",
    "dim",
]


def pad_dataset(ds, max_size):
    """
    Pads the dataset ds to the size max_size, by adding missing values
    at the bottom and right of the dataset.
    If the dataset is already larger than max_size, crops it to max_size.
    """
    # Retrieve the size of the dataset
    sizes = ds.sizes
    size = (sizes["scan"], sizes["pixel"])
    # Compute the padding values
    pad_scan = max_size[0] - size[0]
    pad_pixel = max_size[1] - size[1]
    # Set all -9999.9 values to NaN
    ds = ds.where(ds != -9999.9)
    # Along the scan dimension, pad the dataset with NaN values if pad_scan > 0
    # or crop it if pad_scan < 0
    if pad_scan > 0:
        ds = ds.pad(scan=(0, pad_scan), mode="constant", constant_values=NA)
    elif pad_scan < 0:
        ds = ds.isel(scan=slice(None, max_size[0]))
    # Same for the pixel dimension
    if pad_pixel > 0:
        ds = ds.pad(pixel=(0, pad_pixel), mode="constant", constant_values=NA)
    elif pad_pixel < 0:
        ds = ds.isel(pixel=slice(None, max_size[1]))
    return ds


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


def preprocess_source_files(ds, max_size):
    # Discard the 'angle_bins' variables
    ds = ds.drop_vars(["ScanTime", "angle_bins"])
    # Reverse the dataset spatially if the image was taken on the descending pass
    ds = reverse_spatially(ds)
    return pad_dataset(ds, max_size)


def process_swath(sensat, swath, files, dest_path, cfg, use_cache=True, verbose=True):
    """Computes the normalization constants and spatial size for a given swath.

    Returns:
        mean (xarray.Dataset): the mean of the training set for each band
        std (xarray.Dataset): the standard deviation of the training set for each band
        max_size (tuple): the maximum size of the images from the swath
    """
    sat, sensor = sensat.split("_")
    print(f"{sensat}: processing swath {swath}")
    # Create the destination directory dest_path/under microwave/SENSOR.SATELLITE/swath/
    dest_swath_dir = dest_path / "microwave" / sat / sensor / swath
    dest_swath_dir.mkdir(parents=True, exist_ok=True)
    # Although a swath always contains the same bands, the size of the images is not
    # identical across the samples, due to missing values that have been removed.
    # Therefore, we can't stack the images into a single xarray dataset just yet.
    # To do so, we'll pad all images from the same swath to the same size. That size
    # is the maximum size of all images from the swath.
    if not use_cache or not (dest_swath_dir / "max_size.txt").exists():
        # - Retrieve the maximum size of the images from the swath
        max_size = [0, 0]
        if verbose:
            print("Retrieving the maximum size of the images")
        iterator = tqdm(files) if verbose else files
        for file in iterator:
            with nc.Dataset(file, "r") as ds:
                dims = ds["passive_microwave"][swath].dimensions
                size = (dims["scan"].size, dims["pixel"].size)
                max_size[0] = max(max_size[0], size[0])
                max_size[1] = max(max_size[1], size[1])
        # Write the max_size to a dest_swath_dir/max_size.txt file
        with open(dest_swath_dir / "max_size.txt", "w") as f:
            f.write(f"{max_size[0]},{max_size[1]}")
    else:
        with open(dest_swath_dir / "max_size.txt", "r") as f:
            max_size = list(map(int, f.read().split(",")))

    # Try to load previous normalization values if they exist
    if not use_cache or not (dest_swath_dir / "normalization_mean.nc").exists():
        # Normalization
        # If 'use_subsample' is set to True (or isn't specified), we'll only load a subset of the data
        # for computing the normalization values. Note that the sensors' brightness temperatures are
        # calibrated in TC-PRIMED, and the mean / std are supposedly stationnary, which means that
        # using a subset of the data shouldn't significantly affect the normalization values.
        if "use_subsample" not in cfg or cfg["use_subsample"]:
            n_files = len(files)
            # Use 5% of the files, with a minimum of 100 files
            n_subsample = max(100, int(0.05 * n_files))
            # If the number of files is less than 100, use all the files
            n_subsample = min(n_files, n_subsample)
            files = np.random.choice(files, n_subsample, replace=False)

        dataset = xr.open_mfdataset(
            files,
            concat_dim="sample",
            combine="nested",
            group=f"passive_microwave/{swath}",
            preprocess=lambda ds: preprocess_source_files(ds, max_size),
        )
        # Now, load the overpass_metadata group of the files, which notably include
        # the overpass time, basin, year and storm number.
        meta = (
            xr.open_mfdataset(
                files,
                concat_dim="time",
                combine="nested",
                group="overpass_metadata",
            )
            .rename_dims({"time": "sample"})
            .reset_index("time")
            .reset_coords()
            .load()
        )
        # Compute the mean and standard deviation of the selected data, for
        # each variable.
        mean = dataset.mean(dim=["sample", "scan", "pixel"], skipna=True)
        std = dataset.std(dim=["sample", "scan", "pixel"], skipna=True)
        # To reinitialize the chunking, xarray advises to write the mean/std to a file and reload them.
        # Use the dask diagnostics progress bar to monitor the progress of the computation
        if verbose:
            print("Computing normalization values")
            with ProgressBar():
                mean = mean.compute()
                std = std.compute()
        mean.to_netcdf(dest_swath_dir / "normalization_mean.nc")
        std.to_netcdf(dest_swath_dir / "normalization_std.nc")
        dataset.close()
        meta.close()
    # (Re)load the mean and standard deviation
    mean = xr.open_dataset(dest_swath_dir / "normalization_mean.nc").load()
    std = xr.open_dataset(dest_swath_dir / "normalization_std.nc").load()
    return mean, std, max_size


def process_sensat(sensat, swaths, files, dest_path, cfg, use_cache=True, verbose=True):
    """Computes the normalization constants and spatial size for a given sensor-satellite pair."""
    means, stds, max_sizes = {}, {}, {}
    for swath in swaths:
        mean, std, max_size = process_swath(
            sensat, swath, files, dest_path, cfg, use_cache, verbose
        )
        means[sensat, swath] = mean
        stds[sensat, swath] = std
        max_sizes[sensat, swath] = max_size
    return means, stds, max_sizes


def process_storm(
    season,
    basin,
    number,
    files,
    sen_sat_pairs,
    sen_sat_swaths,
    means,
    stds,
    max_sizes,
    dest_path,
    metadata_path,
    verbose=True,
):
    """For a given storm, loads all files related to it, normalizes all sources, applies
    some preprocessing, and saves the result."""
    sid = f"{season}{basin}{number}"
    if verbose:
        print(f"Processing storm {sid}")
    # Save the storm's data to the directory dest_path/season/basin/sid.nc
    # Each source will be stored in the netcdf group 'SENSOR_SATELLITE_swath'
    storm_dest_dir = dest_path / str(season) / str(basin)
    storm_dest_dir.mkdir(parents=True, exist_ok=True)
    storm_dest_path = storm_dest_dir / f"{sid}.nc"
    # We'll write into the result file in append mode. To make sure we start from scratch,
    # we'll delete the file if it already exists.
    if storm_dest_path.exists():
        storm_dest_path.unlink()
    # We'll process the storm source by source
    for sensat in sen_sat_pairs:
        # Isolate the files corresponding to the sensor-satellite pair
        sen_sat_files = [file for file in files if sensat in file.stem]
        if len(sen_sat_files) == 0:
            continue
        for swath in sen_sat_swaths[sensat]:
            # if not (sensat == "GMI_GPM" and swath == "S2"):
            #   continue
            # Load the files
            dataset = xr.concat(
                [
                    preprocess_source_files(
                        xr.open_dataset(
                            file,
                            group=f"passive_microwave/{swath}",
                        ),
                        max_sizes[sensat, swath],
                    )
                    for file in sen_sat_files
                ],
                dim="sample",
            ).load()
            # Load the storm's metadata
            storm_meta = (
                xr.concat(
                    [
                        xr.open_dataset(
                            file,
                            group="overpass_metadata",
                        )
                        for file in sen_sat_files
                    ],
                    dim="time",
                )
                .rename_dims({"time": "sample"})
                .reset_index("time")
                .reset_coords()
                .load()
            )
            # Sort by time
            storm_meta = storm_meta.sortby("time")
            dataset = dataset.sortby(storm_meta["time"])
            # Drop any encoding, to reduce the loading time of the preprocessed data
            dataset = dataset.drop_encoding()
            # Compute the distance between each pixel and the storm center as a new variable
            # using the 'x' and 'y' variables
            dataset["dist_to_center"] = np.sqrt(dataset["x"] ** 2 + dataset["y"] ** 2)
            dataset = dataset.drop_vars(["x", "y"])
            # Add the land-sea mask as a new variable.
            # We need to convert NaN values to 0, as the land-sea mask function does not handle them.
            # Note: globe.is_land expects the longitude to be in the range [-180, 180]
            lon = np.nan_to_num(dataset.longitude.values)
            lon[lon > 180] -= 360
            mask = globe.is_land(
                np.nan_to_num(dataset.latitude.values),
                lon,
            )
            # Where the latitude and longitude are NaN, set the mask to NaN
            mask[np.isnan(dataset.latitude.values)] = np.nan
            dataset["land_mask"] = (("sample", "scan", "pixel"), mask)
            # Normalize the data, only for the variables that are in means and stds
            # and that are not contextual variables ('latitude', 'longitude', 'land_mask').
            for var in means[sensat, swath].data_vars:
                if var in dataset.data_vars and var not in ["latitude", "longitude", "land_mask"]:
                    dataset[var] = (dataset[var] - means[sensat, swath][var]) / stds[
                        sensat, swath
                    ][var]
            # Save the dataset
            full_source_name = f"tc_primed.microwave.{sensat}.{swath}"
            dataset.to_netcdf(storm_dest_dir / f"{sid}.nc", group=full_source_name, mode="a")
            dataset.close()
            # Append the metadata to the CSV metadata file:
            # SID, time, season, basin, cyclone_number, sat_sensor_swath, "2D"
            metadata = storm_meta[["time", "season", "basin", "cyclone_number"]].to_dataframe()
            metadata["sid"] = [sid] * len(metadata)
            metadata["source_name"] = [full_source_name] * len(metadata)
            metadata["dim"] = ["2D"] * len(metadata)
            metadata = metadata[_METADATA_COL_ORDER_]
            metadata.to_csv(metadata_path, mode="a", header=False)
            storm_meta.close()


@hydra.main(config_path="../conf/", config_name="config", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # Path to the raw dataset
    tc_primed_path = Path(cfg["paths"]["raw_datasets"]) / "tc_primed"
    # Path to where the normalization cosntants and other intermediate results specific
    # to each source will be stored
    sources_path = Path(cfg["paths"]["sources"]) / "tc_primed"
    # Path to where the preprocessed dataset will be stored (as netCDF files)
    dest_path = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to where the metadata will be stored
    metadata_path = Path(cfg["paths"]["metadata"])
    use_cache = cfg["use_cache"] if "use_cache" in cfg else False
    if not use_cache:
        print("Not using cache. Use +use_cache=true to use cache.")
    else:
        print("\033[91mUsing cache. Use +use_cache=false or remove arg to disable cache.\033[0m")
    # Retrieve the list of seaons that are used for the validation and test sets
    val_seasons, test_seasons = (
        cfg["general_settings"]["val_seasons"],
        cfg["general_settings"]["test_seasons"],
    )
    # Retrieve the list of files from the TC-Primed dataset, excluding the validation and test sets
    sen_sat_pairs, sen_sat_files, sen_sat_swaths = list_tc_primed_sources(
        tc_primed_path, exclude_years=val_seasons + test_seasons
    )
    # Compute the normalization values for each swath
    print("Computing normalization constants")
    means, stds, max_sizes = {}, {}, {}
    # Process each sensor-satellite pair in parallel
    if "n_workers" in cfg:
        n_workers = cfg["n_workers"]
        futures = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for sensat in sen_sat_pairs:
                futures.append(
                    executor.submit(
                        process_sensat,
                        sensat,
                        sen_sat_swaths[sensat],
                        sen_sat_files[sensat],
                        sources_path,
                        cfg,
                        use_cache,
                        verbose=False,
                    )
                )
            for future in tqdm(futures):
                mean, std, max_size = future.result()
                means.update(mean)
                stds.update(std)
                max_sizes.update(max_size)
    else:
        for sensat in sen_sat_pairs:
            mean, std, max_size = process_sensat(
                sensat,
                sen_sat_swaths[sensat],
                sen_sat_files[sensat],
                sources_path,
                cfg,
                use_cache,
            )
            means.update(mean)
            stds.update(std)
            max_sizes.update(max_size)
    # Erase the metadata file if it already exists
    if metadata_path.exists():
        metadata_path.unlink()
    # Create the header of the metadata file
    with open(metadata_path, "w") as f:
        f.write(",".join(_METADATA_COL_ORDER_) + "\n")
    # Process the storms
    print("Processing storms")
    storm_files = list_tc_primed_storm_files(tc_primed_path)
    if "n_workers" in cfg:
        n_workers = cfg["n_workers"]
        # Process the storms in parallel, with a progress bar
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    process_storm,
                    season,
                    basin,
                    number,
                    files,
                    sen_sat_pairs,
                    sen_sat_swaths,
                    means,
                    stds,
                    max_sizes,
                    dest_path,
                    metadata_path,
                    verbose=False,
                )
                for (season, basin, number), files in storm_files.items()
            ]
            for future in tqdm(futures):
                future.result()
    else:
        for (season, basin, number), files in tqdm(storm_files.items()):
            process_storm(
                season,
                basin,
                number,
                files,
                sen_sat_pairs,
                sen_sat_swaths,
                means,
                stds,
                max_sizes,
                dest_path,
                metadata_path,
            )


if __name__ == "__main__":
    main()
