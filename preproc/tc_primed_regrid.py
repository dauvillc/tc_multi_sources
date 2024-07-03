"""
Clément Dauvilliers - 23/04/2024.
Preprocesses the TC-Primed dataset by normalizing the data, treating missing values,
and formatting the result into the sources tree.
"""

import hydra
import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
from xarray.backends import NetCDF4DataStore
from global_land_mask import globe
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from preproc.utils import list_tc_primed_sources, list_tc_primed_storm_files
from multi_sources.data_processing.grid_functions import regrid


# Order of the metadata columns in the CSV file
_METADATA_COL_ORDER_ = [
    "sid",
    "time",
    "season",
    "basin",
    "cyclone_number",
    "source_name",
    "dim",
    "data_vars",
]


def preprocess_source_files(ds):
    # Discard the 'angle_bins' variables
    ds = ds.drop_vars(["ScanTime", "angle_bins"])
    # Compute the distance between each pixel and the storm center as a new variable
    # using the 'x' and 'y' variables
    ds["dist_to_center"] = np.sqrt(ds["x"] ** 2 + ds["y"] ** 2)
    ds = ds.drop_vars(["x", "y"])
    return ds


def process_swath(sensat, swath, files, dest_path, cfg, use_cache=True, verbose=True):
    """Computes the spatial area for a given swath of a sensor-satellite pair.

    Returns:
        mean (xarray.Dataset): the mean of the training set for each band
        std (xarray.Dataset): the standard deviation of the training set for each band
        area (tuple of float): Tuple (delta_lon, delta_lat) representing the average area
            covered by the images from the swath.
    """
    sat, sensor = sensat.split("_")
    print(f"{sensat}: processing swath {swath}")
    # Create the destination directory dest_path/under microwave/SENSOR.SATELLITE/swath/
    dest_swath_dir = dest_path / "microwave" / sat / sensor / swath
    dest_swath_dir.mkdir(parents=True, exist_ok=True)
    # If requested in the config, use only a sample of the files for computing the normalization
    # constants
    if "use_subsample" in cfg:
        rng = np.random.default_rng(42)
        files_norm = rng.choice(files, cfg["use_subsample"], replace=False)
    else:
        files_norm = files

    if not use_cache or not (dest_swath_dir / "max_area.txt").exists():
        if verbose:
            print("Processing normalization constants and area")
        # Process each file successively. Since the images don't have the same sizes,
        # we can't just stack them and open all of them with open_mfdataset.
        # Thus, we'll compute the avg area and the normalization constants iteratively.
        avg_area = [0, 0]
        means, stds = [], []
        iterator = tqdm(files) if verbose else files
        for file in iterator:
            ds = nc.Dataset(file)['passive_microwave'][swath]
            # Although a swath always contains the same bands, the areas covered by the images
            # aren't exactly the same (due to the satellite's orbit). We'll compute the average
            # area covered by the images from the swath, i.e.
            # - The average delta_lat between the min and max latitudes
            # - The average delta_lon between the min and max longitudes
            with xr.open_dataset(NetCDF4DataStore(ds)) as ds:
                ds = preprocess_source_files(ds)
                # Compute the area covered by the image
                lat_min, lat_max = ds.latitude.min().values, ds.latitude.max().values
                lon_min, lon_max = ds.longitude.min().values, ds.longitude.max().values
                delta_lat = lat_max - lat_min
                delta_lon = lon_max - lon_min
                avg_area[0] += delta_lon
                avg_area[1] += delta_lat

                # Normalize the data if the file is in the subsample
                if file in files_norm:
                    means.append(ds.mean())
                    stds.append(ds.std())
        avg_area[0] /= len(files)
        avg_area[1] /= len(files)
        # Write the avg_area to a dest_swath_dir/avg_area.txt file
        with open(dest_swath_dir / "avg_area.txt", "w") as f:
            f.write(f"{avg_area[0]},{avg_area[1]}")
        # Write the normalization constants to dest_swath_dir/normalization_mean.nc
        # and dest_swath_dir/normalization_std.nc
        mean = xr.concat(means, dim="sample").mean(dim="sample")
        std = xr.concat(stds, dim="sample").mean(dim="sample")
        mean.to_netcdf(dest_swath_dir / "normalization_mean.nc")
        std.to_netcdf(dest_swath_dir / "normalization_std.nc")
    else:
        with open(dest_swath_dir / "avg_area.txt", "r") as f:
            avg_area = list(map(float, f.read().split(",")))
        mean = xr.open_dataset(dest_swath_dir / "normalization_mean.nc")
        std = xr.open_dataset(dest_swath_dir / "normalization_std.nc")
    return mean, std, avg_area


def process_sensat(sensat, swaths, files, dest_path, cfg, use_cache=True, verbose=True):
    """Computes the normalization constants and spatial size for a given sensor-satellite pair."""
    means, stds, avg_areas = {}, {}, {}
    for swath in swaths:
        mean, std, avg_area = process_swath(
            sensat, swath, files, dest_path, cfg, use_cache, verbose
        )
        means[sensat, swath] = mean
        stds[sensat, swath] = std
        avg_areas[sensat, swath] = avg_area
    return means, stds, avg_areas


def process_storm(
    season,
    basin,
    number,
    files,
    sen_sat_pairs,
    sen_sat_swaths,
    means,
    stds,
    avg_areas,
    target_res_km,
    dest_path,
    metadata_path,
    cfg,
    verbose=True,
):
    """For a given storm, loads all files related to it, normalizes all sources, applies
    some preprocessing, and saves the result."""
    sid = f"{season}{basin}{number}"
    if verbose:
        print(f"Processing storm {sid}")
    # Save the storm's data to the directory dest_path/season/basin/sid
    # For timestep t, the data will be saved as the numpy file
    # dest_path/season/basin/sid/t.npy
    storm_dest_dir = dest_path / str(season) / str(basin) / str(sid)
    storm_dest_dir.mkdir(parents=True, exist_ok=True)
    # We'll process the storm source by source
    for sensat in sen_sat_pairs:
        # Isolate the files corresponding to the sensor-satellite pair
        sen_sat_files = [file for file in files if sensat in file.stem]
        if len(sen_sat_files) == 0:
            continue
        # Process each file successively
        for file in sen_sat_files:
            # Load the storm's metadata
            storm_meta = xr.open_dataset(file, group="overpass_storm_metadata").load()
            # Retrieve the intensity (in knots) of the storm at the time of the overpass.
            # Compare it to the threshold defined in the configuration file. If the intensity
            # is below the threshold, skip the file.
            intensity = storm_meta.intensity.values[0]
            if intensity < cfg["min_intensity_knots"]:
                continue
            # Retrieve the distance to land at the time of the overpass. If the distance is
            # negative (the storm is over land), skip the file.
            dist_to_land = storm_meta.distance_to_land.values[0]
            if (not cfg["include_obs_over_land"]) and dist_to_land < 0:
                continue
            storm_meta.close()
            # Load the file's overpass metadata
            overpass_meta = (
                xr.open_dataset(
                    file,
                    group="overpass_metadata",
                )
                .reset_coords()
                .load()
            )
            time = pd.to_datetime(overpass_meta.time.values[0])
            # Process each swath of the sensor-satellite pair for that file
            for swath in sen_sat_swaths[sensat]:
                dataset = preprocess_source_files(
                    xr.open_dataset(
                        file,
                        group=f"passive_microwave/{swath}",
                    ),
                ).load()
                # Regrid the data to a regular grid with the requested resolution
                dataset = regrid(
                    dataset,
                    target_res_km,
                    avg_areas[sensat, swath],
                )
                # Normalize the data, only for the variables that are in means and stds
                # and that are not contextual variables ('latitude', 'longitude',
                # 'x', 'y'
                data_vars = [
                    var
                    for var in dataset.data_vars
                    if var not in ["latitude", "longitude", "dist_to_center", "x", "y"]
                ]
                for var in means[sensat, swath].data_vars:
                    if var in data_vars:
                        dataset[var] = (dataset[var] - means[sensat, swath][var]) / stds[
                            sensat, swath
                        ][var]
                # Add the land-sea mask as a new variable.
                mask = globe.is_land(
                    dataset["latitude"].values,
                    dataset["longitude"].values,
                )
                dataset["land_mask"] = (("lat", "lon"), mask)
                # Remove 'latitude' and 'longitude' from the coordinates
                # so that they are included by to_dataarray()
                dataset = dataset.reset_coords()
                # Save the data for the current time step, sensor-satellite pair and swath
                # as a numpy file
                # The numpy array should be stacked in the following order:
                # latitude, longitude, land_mask, dist_to_center, band1, band2, ..., bandN
                full_source_name = f"tc_primed.microwave.{sensat}.{swath}"
                dest_file = (
                    storm_dest_dir / f"{time.strftime('%Y%m%d%H%M%S')}-{full_source_name}.npy"
                )
                order = ["latitude", "longitude", "land_mask", "dist_to_center"] + data_vars
                np.save(dest_file, dataset.to_dataarray().reindex(variable=order).values)
                # Append the metadata to the CSV metadata file:
                # SID, time, season, basin, cyclone_number, sat_sensor_swath, "2D"
                metadata = pd.DataFrame(
                    {
                        "sid": [sid],
                        "time": [time],
                        "season": [season],
                        "basin": [basin],
                        "cyclone_number": [number],
                        "source_name": [full_source_name],
                        "dim": ["2D"],
                        "data_vars": [None],
                    }
                )
                # Add the list of data variables to the metadata, so that we'll be
                # able to retrieve their order when loading the data
                metadata.at[0, "data_vars"] = data_vars
                metadata = metadata[_METADATA_COL_ORDER_]
                # Save to json and not csv, as one of the columns is a list
                metadata.to_json(metadata_path, mode="a", orient="records", lines=True)
                overpass_meta.close()


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
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
        cfg["val_seasons"],
        cfg["test_seasons"],
    )
    # Retrieve the list of files from the TC-Primed dataset, excluding the validation and test sets
    sen_sat_pairs, sen_sat_files, sen_sat_swaths = list_tc_primed_sources(
        tc_primed_path, exclude_years=val_seasons + test_seasons
    )
    # Compute the normalization values for each swath
    print("Computing normalization constants")
    means, stds, avg_areas = {}, {}, {}
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
                mean, std, avg_area = future.result()
                means.update(mean)
                stds.update(std)
                avg_areas.update(avg_area)
    else:
        for sensat in sen_sat_pairs:
            mean, std, avg_area = process_sensat(
                sensat,
                sen_sat_swaths[sensat],
                sen_sat_files[sensat],
                sources_path,
                cfg,
                use_cache,
            )
            means.update(mean)
            stds.update(std)
            avg_areas.update(avg_area)
    # Retrieve the target regridding resolution from the configuration file
    target_res_km = cfg["target_resolution_km"]
    # Erase the metadata file if it already exists
    if metadata_path.exists():
        metadata_path.unlink()
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
                    avg_areas,
                    target_res_km,
                    dest_path,
                    metadata_path,
                    cfg,
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
                avg_areas,
                target_res_km,
                dest_path,
                metadata_path,
                cfg,
            )


if __name__ == "__main__":
    main()
