import hydra
import numpy as np
import pandas as pd
import json
import xarray as xr
import dask
from netCDF4 import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict
from dask.distributed import Client


def process_source(train_metadata, source_name, constants_dir, use_fraction, min_files):
    """
    Computes the normalization constants for all samples in a source.
    """
    # Filter the metadata to keep only the samples from the source.
    train_metadata = train_metadata[train_metadata["source_name"] == source_name]
    data_paths = [sample["data_path"] for _, sample in train_metadata.iterrows()]
    new_len = max(int(len(data_paths) * use_fraction), min_files)
    data_paths = np.random.choice(
        data_paths, size=min(new_len, len(data_paths)), replace=False
    ).tolist()
    # Concatenate the data from all samples, ignoring the coordinates as we won't
    # be computing the normalization constants for them here.
    ds = xr.open_mfdataset(
        data_paths,
        combine="nested",
        concat_dim="sample",
        parallel=True,
        coords="minimal",
        compat="override",
        autoclose=True,
    )
    ds = ds.drop_vars(["land_mask", "latitude", "longitude"], errors="ignore")
    if "time" in ds.data_vars:
        ds = ds.drop_vars("time", errors="ignore")
    # Compute the mean and std of the data.
    means = ds.mean(skipna=True).compute()
    stds = ds.std(skipna=True).compute()
    means_dict = {var: float(means[var].values) for var in ds.data_vars}
    stds_dict = {var: float(stds[var].values) for var in ds.data_vars}
    ds.close()

    constants_dir.mkdir(parents=True, exist_ok=True)
    with open(constants_dir / "data_means.json", "w") as f:
        json.dump(means_dict, f)
    with open(constants_dir / "data_stds.json", "w") as f:
        json.dump(stds_dict, f)


def load_merged_metadata(processed_dir):
    """Load and merge metadata from all sources."""
    sources_metadata = {}
    for source_dir in processed_dir.iterdir():
        if source_dir.is_dir():
            metadata_file = source_dir / "source_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    sources_metadata[source_dir.name] = json.load(f)
    return sources_metadata


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    num_workers = 0 if "num_workers" not in cfg else cfg["num_workers"]
    # Option to use only a fraction of the samples to compute the normalization constants,
    # while keeping a minimum number of samples.
    use_fraction = cfg["norm_constants_fraction"]
    min_files = cfg["norm_constants_min_samples"]
    max_mem_per_worker = cfg["max_mem_per_worker"]
    # Path to the preprocessed dataset directory
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the regridded dataset
    regridded_dir = preprocessed_dir / "processed"
    # Path to the constants, where the average areas will be stored.
    constants_dir = preprocessed_dir / "constants"
    # Load the metadata for the training samples.
    samples_metadata = pd.read_json(preprocessed_dir / "train.json", orient="records", lines=True)

    # Create the dask client
    client = Client(
        n_workers=max(num_workers, 1),
        threads_per_worker=1,
        memory_limit=max_mem_per_worker,
        processes=True,
    )

    # Load the metadata for all sources
    source_metadata = load_merged_metadata(regridded_dir)

    # The prepared data directory contains one sub-directory per source.
    source_dirs = [d for d in regridded_dir.iterdir() if d.is_dir()]
    for source_dir in tqdm(source_dirs, desc="Processing sources"):
        print("Processing source ", source_dir.name)
        source_name = source_dir.name
        process_source(
            samples_metadata, source_name, constants_dir / source_name, use_fraction, min_files
        )

    # We know need to compute the normalization constants for the context variables.
    # However, the context variables can be shared across sources (e.g. the observing frequency
    # or the IFOV). Thus, we'll load all of the context variables across all samples and compute
    # the normalization constants at once.
    # sources_metadata[source_name]["context_vars"] contains the context variables for the source.
    # Retrieve the list of context variables that can be found in at least one source.
    context_vars = set()
    for source_name, metadata in source_metadata.items():
        context_vars.update(metadata["context_vars"])
    context_vars = list(context_vars)
    # Isolate the context variables in the samples metadata.
    context_df = samples_metadata[context_vars]
    # Now, a cell of context_df is not directly the value of the variable for that sample,
    # but a dict {channel: value}. We need to explode the DataFrame to get the values.
    means, stds = {}, {}
    for cvar in tqdm(context_vars, desc="Computing constants for context variables"):
        df = context_df[cvar]
        # Not all samples have a value for this context variable, which
        # creates NaN values.
        df = df[~df.isna() & (df != "") & (df != {})]
        df = df.apply(lambda m: list(m.values())).explode()
        means[cvar] = df.mean()
        stds[cvar] = df.std()

    # Save the means and stds in two JSON files.
    json.dump(means, open(constants_dir / "context_means.json", "w"))
    json.dump(stds, open(constants_dir / "context_stds.json", "w"))

    client.close()


if __name__ == "__main__":
    main()
