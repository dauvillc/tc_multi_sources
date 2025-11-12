"""To be used after regrid.py. Splits the data into training, validation, and test sets."""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # Path to the preprocessed dataset
    preprocessed_dir = Path(cfg["paths"]["preprocessed_dataset"])
    # Path to the preprocessed dataset
    regridded_dir = preprocessed_dir / "prepared"
    train_file = preprocessed_dir / "train.csv"
    val_file = preprocessed_dir / "val.csv"
    test_file = preprocessed_dir / "test.csv"
    # Cfg option to reuse existing splits if they exist:
    # if an SID is alreay in one of the splits, we won't change it
    # (allows to add new data without changing existing splits).
    reuse_existing = cfg["reuse_existing_splits"]
    reuse_existing = reuse_existing and (
        train_file.exists() and val_file.exists() and test_file.exists()
    )

    # Each subdirectory in the regridded directory corresponds to a source, and contains
    # a file "samples_metadata.csv". We'll load all of these files and assemble them
    # into a single DataFrame, which we'll then split into training, validation, and test sets.
    metadata = []
    for source_dir in tqdm(list(regridded_dir.iterdir()), desc="Loading metadata"):
        metadata_file = source_dir / "samples_metadata.csv"
        metadata.append(pd.read_csv(metadata_file, parse_dates=["time"]))
    print("Concatenating metadata")
    metadata = pd.concat(metadata, ignore_index=True)

    # In some cases the same source may have multiple images for the same storm
    # very close in time (e.g. at less than an hour interval). Those images would
    # be almost identical, so we'll filter the data to keep only one image every
    # min_time_between_same_source minutes for each storm and source.
    metadata = metadata.sort_values(["sid", "source_name", "time"])
    delta = pd.Timedelta(minutes=cfg["min_time_between_same_source"])

    def keep_group(g):
        kept = []
        last_kept_time = pd.Timestamp.min
        for idx, t in zip(g.index, g["time"]):
            if t >= last_kept_time + delta:
                kept.append(idx)
                last_kept_time = t
        return g.loc[kept]

    metadata = (
        metadata.groupby(["sid", "source_name"], group_keys=False)
        .apply(keep_group)
        .reset_index(drop=True)
    )

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

    # Remove sids that are already in one of the splits if reuse_existing is True
    if reuse_existing:
        print("Reusing existing splits")
        existing_train = pd.read_csv(train_file)
        existing_val = pd.read_csv(val_file)
        existing_test = pd.read_csv(test_file)
        existing_sids = (
            set(existing_train["sid"].unique())
            .union(set(existing_val["sid"].unique()))
            .union(set(existing_test["sid"].unique()))
        )
        print(f"Number of pre-existing storms in splits: {len(existing_sids)}")
        sids = [sid for sid in sids if sid not in existing_sids]
        if len(sids) == 0:
            print("No new storms to split, exiting.")
            return
        print(f"Number of new storms to split: {len(sids)}")

    # Split the sids into training, validation, and test sets
    print("Splitting data")
    train_frac, val_frac, test_frac = train_val_test_fraction
    train_val_frac = train_frac + val_frac
    train_vals_sids, test_sids = train_test_split(sids, test_size=test_frac, random_state=seed)
    train_sids, val_sids = train_test_split(
        train_vals_sids, test_size=val_frac / train_val_frac, random_state=seed
    )
    # Place the samples into the appropriate split based on the storm ID
    train = metadata[metadata["sid"].isin(train_sids)]
    val = metadata[metadata["sid"].isin(val_sids)]
    test = metadata[metadata["sid"].isin(test_sids)]

    # If reusing existing splits, append the new samples to the existing ones
    if reuse_existing:
        print("Appending to existing splits")
        train = pd.concat([existing_train, train], ignore_index=True)
        val = pd.concat([existing_val, val], ignore_index=True)
        test = pd.concat([existing_test, test], ignore_index=True)

    train = train.sort_values(["sid", "time", "source_name"], ascending=[True, False, True])
    val = val.sort_values(["sid", "time", "source_name"], ascending=[True, False, True])
    test = test.sort_values(["sid", "time", "source_name"], ascending=[True, False, True])
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Save the split metadata to disk as preprocessed_dir/train.json, ...
    print("Saving split metadata")
    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Test samples: {len(test)}")


if __name__ == "__main__":
    main()
