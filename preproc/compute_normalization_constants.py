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
            means[var] = np.mean(data)
            stds[var] = np.std(data)
    return means, stds


def process_source(regridded_dir, constants_dir, num_workers=0):
    """
    Computes the normalization constants for all samples in a source.
    """
    source_name = regridded_dir.name
    # Load the samples metadata.
    samples_metadata = pd.read_json(
        regridded_dir / "samples_metadata.json", orient="records", lines=True
    )

    # Compute the normalization constants for the data variables.
    means, stds = [], []
    if num_workers == 0:
        # Process each sample sequentially.
        for _, row in tqdm(
            samples_metadata.iterrows(), desc=f"Computing constants for {source_name}"
        ):
            mean, std = sample_data_constants(row["data_path"])
            means.append(mean)
            stds.append(std)
    else:
        # Process each sample in parallel.
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(sample_data_constants, row["data_path"])
                for _, row in samples_metadata.iterrows()
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
    # Path to the regridded dataset
    regridded_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the constants, where the average areas will be stored.
    constants_dir = Path(cfg["paths"]["constants"])

    # The prepared data directory contains one sub-directory per source.
    source_dirs = [d for d in regridded_dir.iterdir() if d.is_dir()]
    for source_dir in source_dirs:
        process_source(source_dir, constants_dir / source_dir.name, num_workers=num_workers)

    # We know need to compute the normalization constants for the context variables.
    # However, the context variables can be shared across sources (e.g. the observing frequency
    # or the IFOV). Thus, we'll load all of the context variables across all samples and compute
    # the normalization constants at once.
    context_dfs = {}  # map {context_var_name: [dataframes]}
    # - For each source, load the source metadata; the "context_vars" entry contains the names
    # of the context variables for that source. Then, load the samples metadata for that source
    # and extract the context variables to obtain a context DataFrame for that source.
    for source_dir in source_dirs:
        source_metadata = json.load(open(source_dir / "source_metadata.json"))
        context_vars = source_metadata["context_vars"]
        samples_metadata = pd.read_json(
            source_dir / "samples_metadata.json", orient="records", lines=True
        )
        for cvar in context_vars:
            if cvar not in context_dfs:
                context_dfs[cvar] = []
            context_dfs[cvar].append(samples_metadata[cvar])
    # Now, for each context variable found, concatenate all the dataframes and compute the
    # normalization constants.
    means, stds = {}, {}
    for cvar, dfs in tqdm(context_dfs.items(), desc="Computing constants for context variables"):
        df = pd.concat(dfs, axis=0, ignore_index=True)
        # A cell of df is not just a float, it's actually a list of floats, as sources
        # generally contain multiple data variables, and each data variable has its own
        # context. We need to explode the lists and then apply the normalization.
        df = df.explode(cvar)
        means[cvar] = df.mean()
        stds[cvar] = df.std()
    # Save the means and stds in two JSON files.
    json.dump(means, open(constants_dir / "context_means.json", "w"))
    json.dump(stds, open(constants_dir / "context_stds.json", "w"))


if __name__ == "__main__":
    main()
