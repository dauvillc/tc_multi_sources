"""To be used after regrid.py. Splits the data into training, validation, and test sets."""

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # Path to the preprocessed dataset
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the regridded dataset
    regridded_dir = preprocessed_dir / "regridded"

    # Each subdirectory in the regridded directory corresponds to a source, and contains
    # a file "samples_metadata.json". We'll load all of these files and assemble them
    # into a single DataFrame, which we'll then split into training, validation, and test sets.
    metadata = []
    for source_dir in tqdm(list(regridded_dir.iterdir()), desc="Loading metadata"):
        metadata_file = source_dir / "samples_metadata.json"
        metadata.append(pd.read_json(metadata_file, orient="records", lines=True))
    print("Concatenating metadata")
    metadata = pd.concat(metadata, ignore_index=True)

    # We'll sort the samples by storm ID, then by time and finally by source.
    # This isn't required for the pipeline to work, but it makes it easier to
    # inspect the data.
    metadata = metadata.sort_values(["sid", "time", "source_name"], ascending=[True, False, True])
    metadata = metadata.reset_index(drop=True)

    # Load the fraction of samples to use for validation and test sets from the config
    train_val_test_fraction = cfg["train_val_test_fraction"]
    if sum(train_val_test_fraction) != 1:
        raise ValueError("The split fractions must sum to 1")
    # Load the random seed from the config
    seed = cfg["splitting_seed"]
    # the 'sid' column is the storm ID. We'll use this to split the data: samples
    # from the same storm should be in the same split, to avoid data leakage.
    sids = metadata["sid"].unique()
    # Split the sids into training, validation, and test sets
    print("Splitting data")
    train_frac, val_frac, test_frac = train_val_test_fraction
    train_val_frac = train_frac + val_frac
    train_vals_sids, test_sids = train_test_split(sids, test_size=test_frac, random_state=seed)
    train_sids, val_sids = train_test_split(
        sids, test_size=val_frac / train_val_frac, random_state=seed
    )
    # Place the samples into the appropriate split based on the storm ID
    train = metadata[metadata["sid"].isin(train_sids)]
    val = metadata[metadata["sid"].isin(val_sids)]
    test = metadata[metadata["sid"].isin(test_sids)]

    # Save the split metadata to disk as preprocessed_dir/train.json, ...
    print("Saving split metadata")
    train_file = preprocessed_dir / "train.json"
    val_file = preprocessed_dir / "val.json"
    test_file = preprocessed_dir / "test.json"
    train.to_json(train_file, orient="records", lines=True, default_handler=str)
    val.to_json(val_file, orient="records", lines=True, default_handler=str)
    test.to_json(test_file, orient="records", lines=True, default_handler=str)


if __name__ == "__main__":
    main()
