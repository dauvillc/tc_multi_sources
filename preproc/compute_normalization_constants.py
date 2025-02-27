import hydra
import numpy as np
import pandas as pd
import json
import xarray as xr
from netCDF4 import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict


def process_sample(sample_path):
    """Given a path to a netCDF file, computes the mean and std of all variables
    in the Dataset."""
    with Dataset(sample_path, "r") as ds:
        means = {var: np.mean(ds[var][:]) for var in ds.variables}
        stds = {var: np.std(ds[var][:]) for var in ds.variables}
    return means, stds


def process_samples_chunk(paths):
    """Process multiple samples and aggregate their statistics."""
    means_dict, stds_dict = defaultdict(list), defaultdict(list)
    for path in paths:
        means, stds = process_sample(path)
        for var, mean in means.items():
            means_dict[var].append(mean)
        for var, std in stds.items():
            stds_dict[var].append(std)
    return means_dict, stds_dict


def process_source(
    train_metadata, source_name, constants_dir, use_fraction, min_files, num_workers
):
    """
    Computes the normalization constants for all samples in a source.
    """
    # Filter the metadata to keep only the samples from the source.
    train_metadata = train_metadata[train_metadata["source_name"] == source_name]
    # Select a fraction of the samples to compute the normalization constants.
    data_paths = train_metadata.sample(frac=use_fraction)["data_path"].values

    if num_workers > 1:
        # Split the data paths into chunks to be processed by different workers.
        chunks = np.array_split(data_paths, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = executor.map(process_samples_chunk, chunks)
            chunk_results = list(tqdm(futures, total=len(chunks), desc=f"Processing {source_name}"))
        
        # Merge results from all chunks
        means_dict, stds_dict = defaultdict(list), defaultdict(list)
        for chunk_means, chunk_stds in chunk_results:
            for var, means in chunk_means.items():
                means_dict[var].extend(means)
            for var, stds in chunk_stds.items():
                stds_dict[var].extend(stds)
    else:
        means_dict, stds_dict = process_samples_chunk(data_paths)

    # Compute the final means and stds.
    means_dict = {var: str(np.mean(means)) for var, means in means_dict.items()}
    stds_dict = {var: str(np.mean(stds)) for var, stds in stds_dict.items()}

    # Save the means and stds in two JSON files.
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
    process_only = cfg.get("process_only", [])
    # Path to the preprocessed dataset directory
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the regridded dataset
    regridded_dir = preprocessed_dir / "processed"
    # Path to the constants, where the average areas will be stored.
    constants_dir = preprocessed_dir / "constants"
    # Load the metadata for the training samples.
    samples_metadata = pd.read_json(preprocessed_dir / "train.json", orient="records", lines=True)

    # Load the metadata for all sources
    source_metadata = load_merged_metadata(regridded_dir)

    # The prepared data directory contains one sub-directory per source.
    source_dirs = [d for d in regridded_dir.iterdir() if d.is_dir()]
    for source_dir in tqdm(source_dirs, desc="Processing sources"):
        if process_only and source_dir.name not in process_only:
            print("Skipping source ", source_dir.name)
            continue
        print("Processing source ", source_dir.name)
        source_name = source_dir.name
        process_source(
            samples_metadata,
            source_name,
            constants_dir / source_name,
            use_fraction,
            min_files,
            num_workers,
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


if __name__ == "__main__":
    main()
