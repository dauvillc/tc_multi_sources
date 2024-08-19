"""Implements the MultiSourceDataset class."""

import torch
import pandas as pd
import json
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from multi_sources.data_processing.source import Source


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.
    A sample from the dataset is a dict {source_name: (A, S, DT, C, D, V)}, where:
    - A is a scalar tensor of shape (1,) containing 1 if the element is available and -1 otherwise.
    - DT is a scalar tensor of shape (1,) containing the time delta between the reference time
      and the element's time, normalized by dt_max.
    - CT is a tensor of shape (n_context_vars,) containing the context variables.
        Each data variable within a source has its own context variables, which are all
        concatenated into a single tensor to form CT.
    - C is a tensor of shape (2, H, W) containing the latitude and longitude at each pixel.
    - LM is a tensor of shape (H, W) containing the land mask.
    - D is a tensor of shape (H, W) containing the distance to the center of the storm.
    - V is a tensor of shape (n_variables, H, W) containing the variables for the source.

    For a given storm S and reference time t0,
    the dataset returns the element from each source that is closest
    to t0 and between t0 and t0 - dt_max.
    If for a given source, no element is available for the storm S and time t0, the element is
    filled with NaNs.
    """

    def __init__(
        self,
        dataset_dir,
        constants_dir,
        included_variables_dict,
        single_channel_sources=True,
        include_seasons=None,
        exclude_seasons=None,
        reference_times_interval=6,
        dt_max=24,
        min_available_sources_prop=0.6,
        enable_data_augmentation=False,
    ):
        """
        Args:
            dataset_dir (str): The directory containing the preprocessed dataset.
                The directory should have the following structure:
                    dataset_dir/
                        source_1/
                            source_metadata.json
                            samples_metadata.json
                        source_2/
                        ...
                    The source_metadata.json files should contain the metadata for each source,
                    while the samples_metadata.json files should contain the metadata for each
                    sample, including the path.
            constants_dir (str): The directory containing the constants for the dataset:
                constants_dir/
                    context_means.json
                    context_stds.json
                    source_1/
                        data_means.json data_stds.json
            included_variables_dict (dict): A dictionary {source_name: list of variables}
                containing the variables to include for each source. A source not in the
                dictionary will not be included in the dataset.
            single_channel_sources (bool): If True, sources with multiple channels will
                be split into multiple sources with a single channel each.
            include_seasons (list of int): The years to include in the dataset.
                If None, all years are included.
            exclude_seasons (list of int): The years to exclude from the dataset.
                If None, no years are excluded.
            reference_times_interval (int): The interval between the reference times, in hours.
            dt_max (int): The maximum time delta between the elements returned for each source,
                in hours.
            min_available_sources_prop (float): For a given sample (storm/time pair),
                the minimum proportion of sources that must have an available element
                for the sample to be included in the dataset.
            enable_data_augmentation (bool): If True, data augmentation is enabled.
        """
        self.h = reference_times_interval
        self.dt_max = pd.Timedelta(dt_max, unit="h")
        self.dataset_dir = Path(dataset_dir)
        self.variables_dict = included_variables_dict
        self.single_channel_sources = single_channel_sources
        self.enable_data_augmentation = enable_data_augmentation

        print("Browsing requested sources and loading metadata...")
        # For each requested source (key in the variables dict), verify that there is a directory
        # in the dataset with that name.
        self.sources, samples_dfs = [], []
        for source_name in self.variables_dict:
            source_dir = self.dataset_dir / source_name
            if not source_dir.is_dir():
                raise ValueError(f"Source {source_name} not found in the dataset.")
            else:
                source_metadata_path = source_dir / "source_metadata.json"
                # Create a Source object from the metadata
                with open(source_metadata_path, "r") as f:
                    source_metadata = json.load(f)
                    # Check that all variables specified for that source in the config
                    # can actually be found in the dataset.
                    for var in self.variables_dict[source_name]:
                        if var not in source_metadata["data_vars"]:
                            raise ValueError(
                                f"Variable {var} is not available for source {source_name}."
                            )
                    # Only keep the variables requested.
                    source_metadata["data_vars"] = self.variables_dict[source_name]
                    self.sources.append(Source(**source_metadata))
                # Load the samples metadata
                samples_metadata = pd.read_json(
                    source_dir / "samples_metadata.json",
                    orient="records",
                    lines=True,
                    convert_dates=["time"],
                )
                samples_dfs.append(samples_metadata)
        print(f"Found {len(self.sources)} sources in the dataset.")
        # Merge the samples metadata
        print("Merging samples metadata...")
        self.df = pd.concat(samples_dfs, ignore_index=True)

        # Filter the dataframe to only keep the rows where source_name is the name of a source
        # in self.sources
        source_names = [source.name for source in self.sources]
        self.source_variables = {source.name: source.data_vars for source in self.sources}
        self.df = self.df[self.df["source_name"].isin(source_names)]

        # Check that the included and excluded seasons are not overlapping
        if include_seasons is not None and exclude_seasons is not None:
            if set(include_seasons).intersection(exclude_seasons):
                raise ValueError("The included and excluded seasons are overlapping.")
        # Select the seasons to include
        if include_seasons is not None:
            self.df = self.df[self.df["season"].isin(include_seasons)]
        # Select the seasons to exclude
        if exclude_seasons is not None:
            self.df = self.df[~self.df["season"].isin(exclude_seasons)]
        if len(self.df) == 0:
            raise ValueError("No elements available for the selected sources and seasons.")
        # ========================================================================================
        # We'll now create a dataframe self.samples_df that contains the list of unique
        # (sid, ref_time) pairs that will be used to sample the dataset.
        # Some more processing is apply to it, which was first written in the notebook
        # notebooks/sources.ipynb.
        # Compute the reference time that is closest and after the element's time
        self.df["ref_time"] = self.df["time"].dt.ceil(f"{self.h}h")
        # For each element, compute the list of reference times that are within dt_max
        # of the element's time, e.g. for an interval of 6h and a maximum time delta of 24h,
        # 06:41:00 -> [12:00:00, 18:00:00, 00:00:00 (next day), 06:00:00 (next day)]
        self.df["ref_time"] = self.df.apply(
            lambda row: pd.date_range(
                row["ref_time"], row["time"] + self.dt_max, freq=f"{self.h}h"
            ),
            axis=1,
        )
        # Explode the reference times into separate rows
        self.df = self.df.explode("ref_time")
        # Filtering:
        # - Compute for each storm/time the number of available sources
        avail = (
            self.df.groupby(["sid", "ref_time"])["source_name"]
            .nunique()
            .rename("avail_source")
            .reset_index()
        )
        avail["avail_frac"] = avail["avail_source"] / self._get_n_sources()
        # - Keep only the storm/time pairs for which at least min_available_sources_prop sources
        # are available
        avail = avail[avail["avail_frac"] >= min_available_sources_prop]
        self.df = self.df.merge(avail[["sid", "ref_time"]], on=["sid", "ref_time"], how="inner")
        # - Sort by sid,source,time and for every (sid,ref_time,source) triplet,
        # keep only the last one (i.e. the one with the latest time)
        self.df = self.df.sort_values(["sid", "source_name", "time"]).drop_duplicates(
            ["sid", "ref_time", "source_name"], keep="last"
        )
        # Finally, compute the list of unique (sid,ref_time) pairs.
        # Each row of this final dataframe will
        # constitute a sample of the dataset, while self.df can be used
        # to retrieve the corresponding elements.
        self.samples_df = self.df[["sid", "ref_time"]].drop_duplicates().reset_index(drop=True)

    def __getitem__(self, idx):
        """Returns the element at the given index.

        Args:
            idx (int): The index of the element to retrieve.

        Returns:
            sample (multi_sources.data_processing.multi_source_batch.MultiSourceBatch): The element
                at the given index.
        """
        sample = self.samples_df.iloc[idx]
        sid = sample["sid"]
        ref_time = sample["ref_time"]
        output = {}
        source_shapes = self._get_source_shapes()
        # Isolate the rows of self.df corresponding to the given sid and ref_time
        sample_df = self.df[(self.df["sid"] == sid) & (self.df["ref_time"] == ref_time)]
        # For each source, try to load the element at the given time
        for source in self.sources:
            source_name = source.name
            # Try to find an element for the given source, sid and ref_time
            df = sample_df[sample_df["source_name"] == source_name]
            if len(df) == 0 or ref_time - df["time"].iloc[0] > self.dt_max:
                source_shape = source_shapes[source_name]
                context_shape = (self._get_n_context_variables(source_name),)

                # No element available for this source
                # Create a tensor of the appropriate shape filled with NaNs
                A = torch.tensor(-1, dtype=torch.float32)
                source_channels = self._get_n_data_variables(source_name)
                DT = torch.tensor(float("nan"), dtype=torch.float32)
                CT = torch.full(context_shape, fill_value=float("nan"), dtype=torch.float32)
                C = torch.full((2, *source_shape), fill_value=float("nan"), dtype=torch.float32)
                LM = torch.full(source_shape, fill_value=float("nan"), dtype=torch.float32)
                D = torch.full(source_shape, fill_value=float("nan"), dtype=torch.float32)
                # Either consider each variable as a separate source, or keep the source as a
                # multi-channel image.
                if self.single_channel_sources:
                    V = torch.full(
                        (1, *source_shape), fill_value=float("nan"), dtype=torch.float32
                    )
                    for dvar in source.data_vars:
                        output[f"{source_name}_{dvar}"] = (A, DT, CT, C, LM, D, V)
                else:
                    V = torch.full(
                        (source_channels, *source_shape),
                        fill_value=float("nan"),
                        dtype=torch.float32,
                    )
                    output[source_name] = (A, DT, CT, C, LM, D, V)

            else:
                time = df["time"].iloc[0]
                dt = ref_time - time
                A = torch.tensor(1, dtype=torch.float32)
                DT = torch.tensor(
                    dt.total_seconds() / self.dt_max.total_seconds(), dtype=torch.float32
                )

                # Load the npy file containing the data for the given storm, time, and source
                filepath = Path(df["data_path"].iloc[0])
                ds = Dataset(filepath)
                C = np.stack([ds["latitude"][:], ds["longitude"][:]], axis=0)
                C = torch.tensor(C, dtype=torch.float32)
                # Load the land mask and distance to the center of the storm
                LM = torch.tensor(ds["land_mask"][:], dtype=torch.float32)
                D = torch.tensor(ds["dist_to_center"][:], dtype=torch.float32)

                if self.single_channel_sources:
                    for dvar in source.data_vars:
                        # Load the context variables for the given data variable
                        context_df = df[source.context_vars].iloc[0]
                        CT = np.array([context_df[cvar][dvar] for cvar in source.context_vars])
                        CT = torch.tensor(CT, dtype=torch.float32)
                        # Load the data variable, yield it with shape (1, H, W)
                        V = torch.tensor(ds[dvar][:], dtype=torch.float32).unsqueeze(0)
                        output[f"{source_name}_{dvar}"] = (A, DT, CT, C, LM, D, V)
                else:
                    # The context variables can be loaded from the sample dataframe.
                    context_df = df[source.context_vars].iloc[0].values
                    # context_vars[cvar] is a dict {dvar: value} for each data variable dvar
                    # of the source. We want to obtain a tensor of shape
                    # (n_context_vars * n_data_vars,)
                    CT = np.stack(
                        [
                            np.array([context_df[cvar][dvar] for dvar in source.variables])
                            for cvar in source.context_vars
                        ],
                        axis=1,
                    )  # (n_data_vars, n_context_vars)
                    CT = torch.tensor(CT.flatten(), dtype=torch.float32)
                    # Load the variables in the order specified in the source
                    V = np.stack([ds[var][:] for var in source.variables], axis=0)
                    V = torch.tensor(V, dtype=torch.float32)
                    output[source_name] = (A, DT, CT, C, LM, D, V)
        return output

    def __len__(self):
        return len(self.samples_df)

    def _get_source_names(self):
        """Returns a list of the source names (before splitting the sources)."""
        return [source.name for source in self.sources]

    def _get_n_sources(self):
        """Returns the number of original sources (before splitting the sources)."""
        return len(self.sources)

    def _get_source_shapes(self):
        """Returns a dict {source_name: (H, W)} containing the shape of the
        data for each source."""
        return {source.name: source.shape for source in self.sources}

    def _get_n_data_variables(self, source_name=None):
        """Returns either the number of data variables within a source,
        or a dict {source_name: number of data variables}."""
        if source_name is not None:
            return self._get_n_data_variables()[source_name]
        return {source.name: source.n_data_variables() for source in self.sources}

    def _get_n_context_variables(self, source_name=None):
        """Returns either the number of context variables within a source,
        or a dict {source_name: number of context variables}, before splitting the sources."""
        if source_name is not None:
            return self._get_n_context_variables()[source_name]
        return {
            # Each data variable within a source has its own context variables
            source.name: source.n_context_variables() * source.n_data_variables()
            for source in self.sources
        }

    def get_source_names(self):
        """Returns a list of the source names, including the split sources if needed."""
        if self.single_channel_sources:
            return [
                f"{source_name}_{var}"
                for source_name in self._get_source_names()
                for var in self.source_variables[source_name]
            ]
        return self.source_names

    def get_n_sources(self):
        """Returns the number of sources."""
        return len(self.get_source_names())

    def get_n_context_variables(self):
        """Returns a dict {source_name: number of context variables}."""
        if self.single_channel_sources:
            return {
                f"{source.name}_{var}": source.n_context_variables()
                for source in self.sources
                for var in source.variables
            }
        else:
            return self._get_n_context_variables()
