import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm


def process_samples_chunk(paths):
    """Process a chunk of samples, and computes the mean and M2
    statistics for each variable."""
    means_dict, m2_dict = defaultdict(int), defaultdict(int)
    count_dict = defaultdict(int)
    for path in paths:
        with Dataset(path, "r") as ds:
            for var in ds.variables:
                if var in ["latitude", "longitude", "time"]:
                    continue
                count, mean, m2 = count_dict[var], means_dict[var], m2_dict[var]
                data = np.ravel(ds[var][:])
                # If the data contains missing values, the loaded data is a numpy masked array.
                if type(data) is np.ma.core.MaskedArray:
                    data = data.compressed()  # Keeps only the valid values.
                count += len(data)
                delta = data - mean
                mean += np.sum(delta / count)
                delta2 = data - mean
                m2 += np.sum(delta * delta2)
                count_dict[var], means_dict[var], m2_dict[var] = count, mean, m2
    return count_dict, means_dict, m2_dict


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
            chunk_results = list(
                tqdm(futures, total=len(chunks), desc=f"Processing {source_name}")
            )
        # Aggregate the results from all chunks.
        means_dict, sq_dict = defaultdict(int), defaultdict(int)
        count_dict = defaultdict(int)
        for chunk_count, chunk_mean, chunk_m2 in chunk_results:
            for var in chunk_count:
                count, mean, sq = count_dict[var], means_dict[var], sq_dict[var]
                count += chunk_count[var]
                delta = chunk_mean[var] - mean
                mean += delta * chunk_count[var] / count
                delta2 = chunk_mean[var] - mean
                sq += chunk_m2[var] + delta * delta2
                count_dict[var], means_dict[var], sq_dict[var] = count, mean, sq
        stds_dict = {var: np.sqrt(sq_dict[var] / count_dict[var]) for var in sq_dict}
    else:
        count_dict, means_dict, m2_dict = process_samples_chunk(data_paths)
        stds_dict = {var: np.sqrt(m2_dict[var] / count_dict[var]) for var in m2_dict}

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
    process_only = cfg.get("process_only", [])
    # Path to the preprocessed dataset directory
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the regridded dataset
    regridded_dir = preprocessed_dir / "prepared"
    # Path to the constants, where the average areas will be stored.
    constants_dir = preprocessed_dir / "constants"
    # Load the metadata for the training samples.
    samples_metadata = pd.read_csv(preprocessed_dir / "train.csv")

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

    # We now need to compute the normalization constants for the source characteristics.
    # Those are variables that describe sources of the same source type. They are defined
    # for each source in the source metadata, in the "charac_vars" entry. We'll do a min-max
    # normalization of every charac var across all sources of the same source type,
    # for each source type.
    # - Retrieve the source types, and for each type the sources belonging to it.
    src_types = defaultdict(list)
    for src_name, src_meta in source_metadata.items():
        src_types[src_meta["source_type"]].append(src_name)
    # The charac vars can be found under
    # source_metadata[src]["charac_var"][charac_var_name][data_var_name] for each data var.
    # - For each source type, compute the min and max of each charac var across all sources
    #   of the same type.
    for src_type, srcs in src_types.items():
        # Get the charac vars from the first source of the type.
        charac_vars = source_metadata[srcs[0]]["charac_vars"].keys()
        min_max = {charac_var: {"min": np.inf, "max": -np.inf} for charac_var in charac_vars}
        for src in srcs:
            # Check that all charac vars are present in the source metadata.
            if set(charac_vars) != set(source_metadata[src]["charac_vars"].keys()):
                raise ValueError(
                    f"Charac vars for source {src} are not consistent with other\
                     sources of the same type."
                )
            for charac_var, charac_var_data in source_metadata[src]["charac_vars"].items():
                for data_var, data in charac_var_data.items():
                    min_max[charac_var]["min"] = float(
                        min(min_max[charac_var]["min"], np.min(data))
                    )
                    min_max[charac_var]["max"] = float(
                        max(min_max[charac_var]["max"], np.max(data))
                    )
            # Save the min and max for each charac var in that source's constants directory.
            with open(constants_dir / src / "charac_vars_min_max.json", "w") as f:
                json.dump(min_max, f)


if __name__ == "__main__":
    main()
