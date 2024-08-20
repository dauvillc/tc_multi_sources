import hydra
import xarray as xr
import numpy as np
import pandas as pd
import json
from netCDF4 import Dataset
from global_land_mask import globe
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multi_sources.data_processing.grid_functions import regrid, grid_distance_to_point


def sample_area(sample_path):
    """
    Computes the area (as a pair delta_lat, delta_lon) covered by the
    observations in a given sample.
    """
    with Dataset(sample_path) as ds:
        lats = ds["latitude"][:]
        lons = ds["longitude"][:]
        delta_lat = np.max(lats) - np.min(lats)
        delta_lon = np.max(lons) - np.min(lons)
        # Case where the longitude spans the 180th meridian.
        if delta_lon > 180:
            delta_lon = 360 - delta_lon
    return delta_lat, delta_lon


def source_average_area(samples_metadata, num_workers=0):
    """
    Browses all samples of a given source and computes the average latitude and
    longitude delta covered by the observations.
    """
    # Each row in samples_metadata is a sample; the data_path column indicates the
    # path to the actual data.
    total_delta_lat, total_delta_lon = 0, 0
    if num_workers == 0:
        # Compute the area covered by each sample sequentially.
        for _, row in tqdm(
            samples_metadata.iterrows(), desc="Computing average area", total=len(samples_metadata)
        ):
            delta_lat, delta_lon = sample_area(row["data_path"])
            total_delta_lat += delta_lat
            total_delta_lon += delta_lon
    else:
        # Compute the area covered by each sample in parallel.
        with ProcessPoolExecutor(num_workers) as executor:
            futures = [
                executor.submit(sample_area, row["data_path"])
                for _, row in samples_metadata.iterrows()
            ]
            for future in tqdm(futures, total=len(futures), desc="Computing average area"):
                delta_lat, delta_lon = future.result()
                total_delta_lat += delta_lat
                total_delta_lon += delta_lon
    # Compute the average area.
    num_samples = len(samples_metadata)
    average_delta_lat = total_delta_lat / num_samples
    average_delta_lon = total_delta_lon / num_samples
    return average_delta_lat, average_delta_lon


def process_sample(sample, average_area, target_resolution, dest_dir):
    """
    Processes a single sample.
    """
    sample_path = sample["data_path"]
    with xr.open_dataset(sample_path) as ds:
        # The regridding function expects (lon, lat) instead of (lat, lon).
        target_resolution = target_resolution[::-1]
        average_area = average_area[::-1]
        # First regrid the sample.
        ds = regrid(ds, target_resolution, average_area)
        for variable in ds.variables:
            if ds[variable].isnull().all():
                return
        # Add the land-sea mask as a new variable.
        land_mask = globe.is_land(
            ds["latitude"].values,
            ds["longitude"].values,
        )
        ds["land_mask"] = (("lat", "lon"), land_mask)
        # Compute the distance between each grid point and the center of the storm.
        storm_lat = sample["storm_latitude"]
        storm_lon = sample["storm_longitude"]
        dist_to_center = grid_distance_to_point(
            ds["latitude"].values,
            ds["longitude"].values,
            storm_lat,
            storm_lon,
        )
        ds["dist_to_center"] = (("lat", "lon"), dist_to_center)
        # Save the regridded sample as netCDF file.
        dest_file = dest_dir / f"{Path(sample_path).stem}.nc"
        ds.to_netcdf(dest_file)

        # Return the spatial shape (H, W) of the regridded sample.
        return ds["latitude"].shape[-2:]


def regrid_source(source_dir, regridded_dir, average_area, target_resolution, num_workers=0):
    """
    Regrids all samples of a given source and saves the regridded samples in
    regridded_dir.
    """
    # Load the samples metadata.
    samples_metadata = pd.read_json(
        source_dir / "samples_metadata.json", orient="records", lines=True
    )
    # Create the regridded source directory.
    regridded_source_dir = regridded_dir / source_dir.name
    regridded_source_dir.mkdir(parents=True, exist_ok=True)

    if num_workers == 0:
        # Process each sample sequentially.
        for _, row in tqdm(samples_metadata.iterrows(), desc=f"Regridding {source_dir.name}"):
            img_shape = process_sample(row, average_area, target_resolution, regridded_source_dir)
    else:
        # Process each sample in parallel.
        with ProcessPoolExecutor(num_workers) as executor:
            futures = [
                executor.submit(
                    process_sample,
                    row,
                    average_area,
                    target_resolution,
                    regridded_source_dir,
                )
                for _, row in samples_metadata.iterrows()
            ]
            for future in tqdm(futures, total=len(futures), desc=f"Regridding {source_dir.name}"):
                img_shape = future.result()

    # We'll now create an updated samples_metadata.json file that contains the paths
    # to the regridded samples, and store it in the regridded source directory.
    print(f"Updating samples metadata for {source_dir.name}")
    regridded_samples_metadata = samples_metadata.copy()
    regridded_samples_metadata["data_path"] = regridded_samples_metadata["data_path"].apply(
        lambda x: regridded_source_dir / f"{Path(x).stem}.nc"
    )
    regridded_samples_metadata.to_json(
        regridded_source_dir / "samples_metadata.json",
        orient="records",
        lines=True,
        default_handler=str,
    )
    # Finally, we'll return the shape of the regridded samples.
    return img_shape


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    num_workers = 0 if "num_workers" not in cfg else cfg["num_workers"]
    use_cache = False if "use_cache" not in cfg else cfg["use_cache"]
    # Path to the preprocessed dataset directory.
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the prepared dataset (output of the first step)
    prepared_dir = preprocessed_dir / "prepared"
    # Path to the regridded dataset (output of this step)
    regridded_dir = preprocessed_dir / "regridded"
    # Path to the constants, where the average areas will be stored.
    constants_dir = preprocessed_dir / "constants"

    # The prepared data directory contains one sub-directory per source.
    # Each sub-directory contains samples_metadata.json, source_metadata.json,
    # and the samples themselves as .nc files.
    # - Retrieve the list of sources.
    source_dirs = [d for d in prepared_dir.iterdir() if d.is_dir()]
    # - Compute the average area covered by the observations of each source.
    average_areas = {}
    for source_dir in source_dirs:
        # - Save the average area to a constants_dir / source / average_area.json
        average_area_path = constants_dir / source_dir.name / "average_area.json"
        average_area_path.parent.mkdir(parents=True, exist_ok=True)
        # Check if the average area is already computed and cached.
        if use_cache and average_area_path.exists():
            print(f"Found cached average area for source {source_dir.name}")
            with open(average_area_path, "r") as f:
                average_areas[source_dir.name] = json.load(f)["average_area"]
                continue
        print(f"Computing average area for source {source_dir.name}")
        samples_metadata = pd.read_json(
            source_dir / "samples_metadata.json", orient="records", lines=True
        )
        average_areas[source_dir.name] = source_average_area(samples_metadata, num_workers)
        with open(average_area_path, "w") as f:
            json.dump({"average_area": average_areas[source_dir.name]}, f)

    # - Regrid each source and get the shape of the regridded samples.
    # After regridding a source, load its source_metadata.json file in the
    # regridded dir and save it into a common sources_metadata dict.
    sources_metadata = {}
    for source_dir in source_dirs:
        source_name = source_dir.name
        print(f"Regridding source {source_name}")
        average_area = average_areas[source_name]
        shape = regrid_source(
            source_dir, regridded_dir, average_area, cfg["target_resolution_km"], num_workers
        )
        # Load the source metadata and store it in sources_metadata.
        with open(source_dir / "source_metadata.json", "r") as f:
            source_metadata = json.load(f)
            sources_metadata[source_name] = source_metadata
            sources_metadata[source_name]["shape"] = shape
    # - Save the sources_metadata dict as preprocessed_dir / sources_metadata.json
    with open(preprocessed_dir / "sources_metadata.json", "w") as f:
        json.dump(sources_metadata, f)


if __name__ == "__main__":
    main()
