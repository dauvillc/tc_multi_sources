import warnings
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
from multi_sources.data_processing.grid_functions import (
    regrid,
    grid_distance_to_point,
    ResamplingError,
)


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


def source_average_area(samples_metadata, path):
    """
    Browses all samples of a given source and computes the average latitude and
    longitude delta covered by the observations.
    """
    # Each row in samples_metadata is a sample; the data_path column indicates the
    # path to the actual data.
    total_delta_lat, total_delta_lon = 0, 0
    # Compute the area covered by each sample sequentially.
    for _, row in samples_metadata.iterrows():
        delta_lat, delta_lon = sample_area(row["data_path"])
        total_delta_lat += delta_lat
        total_delta_lon += delta_lon
    # Compute the average area.
    num_samples = len(samples_metadata)
    average_delta_lat = total_delta_lat / num_samples
    average_delta_lon = total_delta_lon / num_samples
    with open(path, "w") as f:
        json.dump({"average_area": (average_delta_lat, average_delta_lon)}, f)
    return average_delta_lat, average_delta_lon


def process_sample(
    sample,
    average_area=None,
    target_resolution=None,
    dest_dir=None,
    dim=1,
    source_name=None,
    no_regrid=None,
):
    """
    Processes a single sample.

    If source is 2D and not in no_regrid list, performs regridding using average_area and target_resolution.
    Otherwise, only applies additional processing (land mask, distance to center).
    Converts 1D coordinate arrays to 2D meshgrids if necessary.
    """
    sample_path = sample["data_path"]
    with xr.open_dataset(sample_path) as ds:
        try:
            # Check if coordinates need to be converted to meshgrid
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            if lats.ndim == 1 and lons.ndim == 1 and dim == 2:
                # Convert 1D coordinate arrays to 2D meshgrids
                lons_2d, lats_2d = np.meshgrid(lons, lats)
                # Replace the 1D coordinates with 2D versions
                ds = ds.assign_coords(
                    latitude=(("lat", "lon"), lats_2d), longitude=(("lat", "lon"), lons_2d)
                )

            # Apply regridding for 2D sources not in no_regrid
            if dim == 2 and (no_regrid is None or source_name not in no_regrid):
                if average_area is not None and target_resolution is not None:
                    ds = regrid(ds, target_resolution, average_area)

            # Rename dimensions if needed. The datasets returned by regrid have
            # "lat" and "lon" as dimensions.
            if "latitude" in ds.dims and "longitude" in ds.dims:
                ds = ds.rename_dims({"latitude": "lat", "longitude": "lon"})

            for variable in ds.variables:
                if ds[variable].isnull().all():
                    warnings.warn(f"Variable {variable} is null for sample {sample_path}.")

            # Compute the land-sea mask
            land_mask = globe.is_land(
                ds["latitude"].values,
                ds["longitude"].values,
            )
            # Compute the distance between each grid point and the center of the storm.
            storm_lat = sample["storm_latitude"]
            storm_lon = sample["storm_longitude"]
            dist_to_center = grid_distance_to_point(
                ds["latitude"].values,
                ds["longitude"].values,
                storm_lat,
                storm_lon,
            )
            # Add the land mask and distance to center as new variables.
            if dim == 2:
                ds["land_mask"] = (("lat", "lon"), land_mask)
                ds["dist_to_center"] = (("lat", "lon"), dist_to_center)
            elif dim == 0:
                ds["land_mask"] = land_mask
                ds["dist_to_center"] = dist_to_center

        except ResamplingError as e:
            print(f"Resampling error for sample {sample_path}: {e}")
            print(f"Row: {sample}")
            return None

        if dest_dir is not None:
            # Save the processed sample as netCDF file.
            dest_file = dest_dir / f"{Path(sample_path).stem}.nc"
            ds.to_netcdf(dest_file)

        # Write the metadata for the sample
        sample["data_path"] = str(dest_file)
        sample = pd.DataFrame([sample])
        sample.to_json(
            dest_dir / "samples_metadata.json",
            orient="records",
            lines=True,
            mode="a",
            default_handler=str,
        )

        # Return the spatial shape
        # --> (H, W) for 2D sources
        if dim == 2:
            return ds["latitude"].shape[-2:]
        # (1,) for 0D sources
        elif dim == 0:
            return (1,)


def process_source_chunk(
    chunk_data, average_area, target_resolution, processed_source_dir, dim, source_name, no_regrid
):
    """
    Process a chunk of samples from a source.
    """
    img_shape = None
    for _, row in chunk_data.iterrows():
        img_shape = process_sample(
            row, average_area, target_resolution, processed_source_dir, dim, source_name, no_regrid
        )
    return img_shape


def process_source(
    source_dir,
    processed_dir,
    average_area=None,
    target_resolution=None,
    num_workers=1,
    dim=1,
    no_regrid=None,
    source_metadata=None,
):
    """
    Processes all samples of a given source, optionally in parallel.
    """
    # Load the samples metadata
    samples_metadata = pd.read_json(
        source_dir / "samples_metadata.json", orient="records", lines=True
    )

    # Create the processed source directory
    processed_source_dir = processed_dir / source_dir.name
    processed_source_dir.mkdir(parents=True, exist_ok=True)

    # Reset the samples metadata file for the processed source
    samples_metadata_path = processed_source_dir / "samples_metadata.json"
    if samples_metadata_path.exists():
        samples_metadata_path.unlink()

    img_shape = None
    if num_workers <= 1:
        # Process samples sequentially
        for _, row in samples_metadata.iterrows():
            img_shape = process_sample(
                row,
                average_area,
                target_resolution,
                processed_source_dir,
                dim,
                source_dir.name,
                no_regrid,
            )
    else:
        # Split samples into chunks for parallel processing
        chunks = np.array_split(samples_metadata, num_workers)

        # Process chunks in parallel
        with ProcessPoolExecutor(num_workers) as executor:
            futures = [
                executor.submit(
                    process_source_chunk,
                    chunk,
                    average_area,
                    target_resolution,
                    processed_source_dir,
                    dim,
                    source_dir.name,
                    no_regrid,
                )
                for chunk in chunks
            ]
            # Get the image shape from any completed future
            for future in tqdm(futures, desc=f"Processing chunks for {source_dir.name}"):
                shape = future.result()
                if shape is not None:
                    img_shape = shape

    # Update and save source metadata
    if source_metadata is not None:
        source_metadata["shape"] = img_shape
        with open(processed_source_dir / "source_metadata.json", "w") as f:
            json.dump(source_metadata, f)

    return img_shape


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    num_workers = 0 if "num_workers" not in cfg else cfg["num_workers"]
    use_cache = False if "use_cache" not in cfg else cfg["use_cache"]
    no_regrid = cfg.get("no_regrid", [])  # Get the no_regrid list from config
    process_only = cfg.get("process_only", None)  # Get the process_only source name
    ignore_source = cfg.get("ignore_source", [])  # Get the ignore_source list from config

    # Path to the preprocessed dataset directory.
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    prepared_dir = preprocessed_dir / "prepared"
    processed_dir = preprocessed_dir / "processed"
    constants_dir = preprocessed_dir / "constants"

    # Load all sources metadata at the beginning
    sources_metadata = {}
    source_dirs = [d for d in prepared_dir.iterdir() if d.is_dir()]
    if process_only:
        source_dirs = [d for d in source_dirs if d.name == process_only]
        if not source_dirs:
            raise ValueError(f"Source {process_only} not found in {prepared_dir}")

    # Filter out ignored sources
    source_dirs = [d for d in source_dirs if d.name not in ignore_source]

    for source_dir in source_dirs:
        with open(source_dir / "source_metadata.json", "r") as f:
            sources_metadata[source_dir.name] = json.load(f)

    # ======================= AVERAGE AREA COMPUTATION =======================
    # - Compute the average area covered by the observations of each source.
    average_areas = {}
    # If num_workers > 0, do it in parallel.
    if num_workers > 0:
        futures = {}
        executor = ProcessPoolExecutor(num_workers)
    for source_dir in source_dirs:
        source_name = source_dir.name
        dim = sources_metadata[source_name].get("dim", 1)
        # Skip average area computation only for 0D sources or sources in no_regrid
        if dim == 0 or source_name in no_regrid:
            continue

        # - Save the average area to a constants_dir / source / average_area.json
        average_area_path = constants_dir / source_name / "average_area.json"
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
        if num_workers <= 1:
            average_area = source_average_area(samples_metadata, average_area_path)
            average_areas[source_dir.name] = average_area
        else:
            futures[source_dir.name] = executor.submit(
                source_average_area, samples_metadata, average_area_path
            )
    if num_workers > 0:
        for source_name, future in tqdm(
            futures.items(), desc="Computing average areas", total=len(futures)
        ):
            average_area = future.result()
            average_areas[source_name] = average_area

    # ======================= PROCESSING SOURCES =======================
    # - Process each source and get the shape of the processed samples.
    shapes = {}
    for source_dir in source_dirs:
        source_name = source_dir.name
        print(f"Processing source {source_name}")
        dim = sources_metadata[source_name].get("dim", 1)
        average_area = average_areas.get(source_name, None)
        shapes[source_name] = process_source(
            source_dir,
            processed_dir,
            average_area,
            cfg["target_resolution"],
            num_workers,
            dim=dim,
            no_regrid=no_regrid,
            source_metadata=sources_metadata[source_name],
        )


if __name__ == "__main__":
    main()
