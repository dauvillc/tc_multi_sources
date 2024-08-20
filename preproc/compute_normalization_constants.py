import hydra
import numpy as np
import pandas as pd
import json
from netCDF4 import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def sample_data_constants(data_path):
    """
    Computes the normalization constants for a single sample.
    """
    # The data is stored in a NetCDF file. We'll return the means and stds as a dict
    # {variable_name: (mean, std)}.
    means, stds = {}, {}
    with Dataset(data_path, "r") as ds:
        # Compute the stats for all variables except 'land_mask', 'latitude'
        # and 'longitude'.
        data_vars = [v for v in ds.variables if v not in ["land_mask", "latitude", "longitude"]]
        for var in data_vars:
            data = ds[var][:]
            means[var] = np.nanmean(data).astype(np.float64)
            stds[var] = np.nanstd(data).astype(np.float64)
    return means, stds


def process_source(train_metadata, source_name, constants_dir, num_workers=0):
    """
    Computes the normalization constants for all samples in a source.
    """
    # Filter the metadata to keep only the samples from the source.
    train_metadata = train_metadata[train_metadata["source_name"] == source_name]
    # Compute the normalization constants for the data variables.
    means, stds = [], []
    if num_workers == 0:
        # Process each sample sequentially.
        for _, row in tqdm(
            train_metadata.iterrows(), desc=f"Computing constants for {source_name}"
        ):
            mean, std = sample_data_constants(row["data_path"])
            means.append(mean)
            stds.append(std)
    else:
        # Process each sample in parallel.
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(sample_data_constants, row["data_path"])
                for _, row in train_metadata.iterrows()
            ]
            for future in tqdm(futures, desc=f"Computing constants for {source_name}"):
                mean, std = future.result()
                means.append(mean)
                stds.append(std)
    # Compute the average means and stds.
    means = {var: np.mean([m[var] for m in means]) for var in means[0]}
    stds = {var: np.mean([s[var] for s in stds]) for var in stds[0]}
    # Save the means and stds in a JSON file.
    with open(constants_dir / "data_means.json", "w") as f:
        json.dump(means, f)
    with open(constants_dir / "data_stds.json", "w") as f:
        json.dump(stds, f)


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    num_workers = 0 if "num_workers" not in cfg else cfg["num_workers"]
    # Path to the preprocessed dataset directory
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the regridded dataset
    regridded_dir = preprocessed_dir / "regridded"
    # Path to the constants, where the average areas will be stored.
    constants_dir = preprocessed_dir / "constants"
    # Load the metadata for the training samples.
    samples_metadata = pd.read_json(preprocessed_dir / "train.json", orient="records", lines=True)

    # The prepared data directory contains one sub-directory per source.
    source_dirs = [d for d in regridded_dir.iterdir() if d.is_dir()]
    for source_dir in source_dirs:
        process_source(
            samples_metadata,
            source_dir.name,
            constants_dir / source_dir.name,
            num_workers=num_workers,
        )

    # We know need to compute the normalization constants for the context variables.
    # However, the context variables can be shared across sources (e.g. the observing frequency
    # or the IFOV). Thus, we'll load all of the context variables across all samples and compute
    # the normalization constants at once.
    # Load the sources metadata.
    with open(preprocessed_dir / "sources_metadata.json", "r") as f:
        source_metadata = json.load(f)
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
        df = df[~df.isna()]
        df = df.apply(lambda m: list(m.values())).explode()
        means[cvar] = df.mean()
        stds[cvar] = df.std()

    # Save the means and stds in two JSON files.
    json.dump(means, open(constants_dir / "context_means.json", "w"))
    json.dump(stds, open(constants_dir / "context_stds.json", "w"))


if __name__ == "__main__":
    main()
