"""Implements the MultiSourceDataset class."""

import torch
import pandas as pd
import json
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from multi_sources.data_processing.source import Source
from multi_sources.data_processing.utils import compute_sources_availability


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.
    A sample from the dataset is a dict {source_name: map}, where each map contains the
    following key-value pairs:
    - "source_type" is a string containing the type of the source.
    - "avail" is a scalar tensor of shape (1,) containing 1 if the element is available
        and -1 otherwise.
    - "dt" is a scalar tensor of shape (1,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "context" is a tensor of shape (n_context_vars,) containing the context variables.
        Each data variable within a source has its own context variables, which are all
        concatenated into a single tensor to form CT.
    - "coords" is a tensor of shape (2, H, W) containing the latitude and longitude at each pixel.
    - "landmask" is a tensor of shape (H, W) containing the land mask.
    - "dist_to_center" is a tensor of shape (H, W) containing the distance
        to the center of the storm.
    - "values" is a tensor of shape (n_variables, H, W) containing the variables for the source.

    For a given storm S and reference time t0,
    the dataset returns the element from each source that is closest
    to t0 and between t0 and t0 - dt_max.
    If for a given source, no element is available for the storm S and time t0, the element is
    filled with NaNs.
    """

    def __init__(
        self,
        dataset_dir,
        split,
        included_variables_dict,
        single_channel_sources=True,
        dt_max=24,
        min_available_sources=2,
        num_workers=0,
        include_seasons=None,
        exclude_seasons=None,
        enable_data_augmentation=False,
    ):
        """
        Args:
            dataset_dir (str): The directory containing the preprocessed dataset.
                The directory should contain the following:
                - train/val/test.json: pandas DataFrame containing the samples metadata.
                - sources_metadata.json: a dictionary containing the metadata for each source.
                - constants/
                    - context_means/stds.json: dictionaries containing the means and stds
                        for the context variables.
                    - source_name/
                        - data_means/stds.json: dictionaries containing the means and stds
                            for the data variables of the source.
            split (str): The split of the dataset to use. Must be one of "train", "val", or "test".
            included_variables_dict (dict): A dictionary {source_name: list of variables}
                containing the variables to include for each source. A source not in the
                dictionary will not be included in the dataset.
            single_channel_sources (bool): If True, sources with multiple channels will
                be split into multiple sources with a single channel each.
            dt_max (int): The maximum time delta between the elements returned for each source,
                in hours.
            min_available_sources (int): The minimum number of sources that must be available
                for a sample to be included in the dataset.
            num_workers (int): If > 1, number of workers to use for parallel loading of the data.
            include_seasons (list of int): The years to include in the dataset.
                If None, all years are included.
            exclude_seasons (list of int): The years to exclude from the dataset.
                If None, no years are excluded.
            enable_data_augmentation (bool): If True, data augmentation is enabled.
        """
        self.dataset_dir = Path(dataset_dir)
        self.constants_dir = self.dataset_dir / "constants"
        self.split = split
        self.variables_dict = included_variables_dict
        self.dt_max = pd.Timedelta(dt_max, unit="h")
        self.single_channel_sources = single_channel_sources
        self.enable_data_augmentation = enable_data_augmentation

        print("Browsing requested sources and loading metadata...")
        # Load the sourcs metadata
        with open(self.dataset_dir / "sources_metadata.json", "r") as f:
            sources_metadata = json.load(f)
        self.sources = []
        for source_name in self.variables_dict:
            if source_name not in sources_metadata:
                raise ValueError(f"Source {source_name} not found in the sources metadata.")
            else:
                # Isolate the variables to include for the source
                sources_metadata[source_name]["data_vars"] = self.variables_dict[source_name]
                self.sources.append(Source(**sources_metadata[source_name]))
        # Load the samples metadata based on the split
        self.df = pd.read_json(
            self.dataset_dir / f"{split}.json",
            orient="records",
            lines=True,
            convert_dates=["time"],
        )
        # Make sure that every source indicated in the config appears
        # in at least one row of the dataset
        unique_sources = self.df["source_name"].unique()
        for source in self.sources:
            if source.name not in unique_sources:
                raise ValueError(f"Source {source.name} is not present in the dataset.")
        print(f"Found {len(self.sources)} sources in the dataset.")

        # Filter the dataframe to only keep the rows where source_name is the name of a source
        # in self.sources
        source_names = [source.name for source in self.sources]
        self.source_variables = {source.name: source.data_vars for source in self.sources}
        self.df = self.df[self.df["source_name"].isin(source_names)]
        self.df = self.df.reset_index(drop=True)

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
        # We'll now filter the samples to only keep the ones where at least min_available_sources
        # sources are available.
        # - We'll compute a dataframe D of shape (n_samples, n_sources) such that
        # D[i, s] = 1 if source i is available for sample s, and 0 otherwise.
        print("Computing sources availability...")
        self.available_sources = compute_sources_availability(
            self.df, self.dt_max, num_workers=num_workers
        )
        # - From this, we can filter the samples to only keep the ones where at least
        # min_available_sources sources are available.
        available_sources_count = self.available_sources.sum(axis=1)
        mask = available_sources_count >= min_available_sources
        self.df = self.df[mask].reset_index(drop=True)
        self.available_sources = self.available_sources[mask].reset_index(drop=True)

        # ========================================================================================
        # Load the data means and stds
        self.data_means, self.data_stds = {}, {}
        for source in self.sources:
            with open(self.constants_dir / source.name / "data_means.json", "r") as f:
                self.data_means[source.name] = json.load(f)
            with open(self.constants_dir / source.name / "data_stds.json", "r") as f:
                self.data_stds[source.name] = json.load(f)

        # Load the context means and stds
        with open(self.constants_dir / "context_means.json", "r") as f:
            self.context_means = json.load(f)
        with open(self.constants_dir / "context_stds.json", "r") as f:
            self.context_stds = json.load(f)

    def __getitem__(self, idx):
        """Returns the element at the given index.

        Args:
            idx (int): The index of the element to retrieve.

        Returns:
            sample (multi_sources.data_processing.multi_source_batch.MultiSourceBatch): The element
                at the given index.
        """
        source_shapes = self._get_source_shapes()
        sample = self.df.iloc[idx]
        sid, t0 = sample["sid"], sample["time"]
        # Isolate the rows of self.df corresponding to the sample sid
        sample_df = self.df[self.df["sid"] == sid]

        # For each source, try to load the element at the given time
        output = {}
        for source in self.sources:
            source_name = source.name
            source_type = source.type
            # Isolate the rows of sample_df corresponding to the right source
            df = sample_df[sample_df["source_name"] == source_name]
            # Sort by descending time so that the first row is the closest to t0
            df = df.sort_values("time", ascending=False)
            # Check that there is at least one element available for this source,
            # and that the time delta is within the acceptable range
            if len(df) == 0 or t0 - df["time"].iloc[0] >= self.dt_max:
                source_shape = source_shapes[source_name]

                # No element available for this source
                # Create a tensor of the appropriate shape filled with NaNs
                A = torch.tensor(-1, dtype=torch.float32)
                source_channels = self._get_n_data_variables(source_name)
                DT = torch.tensor(float("nan"), dtype=torch.float32)
                C = torch.full((2, *source_shape), fill_value=float("nan"), dtype=torch.float32)
                LM = torch.full(source_shape, fill_value=float("nan"), dtype=torch.float32)
                D = torch.full(source_shape, fill_value=float("nan"), dtype=torch.float32)
                # Either consider each variable as a separate source, or keep the source as a
                # multi-channel image.
                if self.single_channel_sources:
                    CT = torch.full(
                        (self._get_n_context_variables(source_name),),
                        fill_value=float("nan"),
                        dtype=torch.float32,
                    )
                    V = torch.full(
                        (1, *source_shape), fill_value=float("nan"), dtype=torch.float32
                    )
                    for dvar in source.data_vars:
                        output[f"{source_name}_{dvar}"] = {
                            "source_type": source_type,
                            "avail": A,
                            "dt": DT,
                            "context": CT,
                            "coords": C,
                            "landmask": LM,
                            "dist_to_center": D,
                            "values": V,
                        }
                else:
                    # If the source is multi-channel, each channel brings its own context variables
                    # so the number of context vars is multiplied by the number of channels.
                    CT = torch.full(
                        (source.n_context_variables() * self._get_n_data_variables(source_name),),
                        fill_value=float("nan"),
                        dtype=torch.float32,
                    )
                    V = torch.full(
                        (source_channels, *source_shape),
                        fill_value=float("nan"),
                        dtype=torch.float32,
                    )
                    output[source_name] = {
                        "source_type": source_type,
                        "avail": A,
                        "dt": DT,
                        "context": CT,
                        "coords": C,
                        "landmask": LM,
                        "dist_to_center": D,
                        "values": V,
                    }

            else:
                time = df["time"].iloc[0]
                dt = t0 - time
                A = torch.tensor(1, dtype=torch.float32)
                DT = torch.tensor(
                    dt.total_seconds() / self.dt_max.total_seconds(), dtype=torch.float32
                )

                # Load the npy file containing the data for the given storm, time, and source
                filepath = Path(df["data_path"].iloc[0])
                with Dataset(filepath) as ds:
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
                            # Normalize the context and values tensors
                            CT, V = self.normalize(CT, V, source, dvar)
                            output[f"{source_name}_{dvar}"] = {
                                "source_type": source_type,
                                "avail": A,
                                "dt": DT,
                                "context": CT,
                                "coords": C,
                                "landmask": LM,
                                "dist_to_center": D,
                                "values": V,
                            }
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
                        # Normalize the context and values tensors
                        CT, V = self.normalize(CT, V, source)
                        output[source_name] = {
                            "source_type": source_type,
                            "avail": A,
                            "dt": DT,
                            "context": CT,
                            "coords": C,
                            "landmask": LM,
                            "dist_to_center": D,
                            "values": V,
                        }
        return output

    def normalize(self, context, values, source, dvar=None):
        """Normalizes the context and values tensors associated with a given
        source, and optionally a specific data variable.
        Args:
            context (torch.Tensor): tensor of shape (n_context_vars,)
            values (torch.Tensor): tensor of shape (C, H, W) if dvar is None,
                or (1, H, W) if dvar is specified.
            source (str): Source object representing the source.
            dvar (str, optional): Name of a specific variable (ie channel) within
                the source to normalize.
        Returns:
            normalized_context (torch.Tensor): normalized context.
            normalized_values (torch.Tensor): normalized values.
        """
        source_name = source.name
        # Context normalization
        context_means = np.array([self.context_means[cvar] for cvar in source.context_vars])
        context_stds = np.array([self.context_stds[cvar] for cvar in source.context_vars])
        # If we're yielding multi-channel sources, then the context tensor is a concatenation
        # of the context variables for each data variable. Thus we need to load
        # the means and stds for each context variable and repeat them for each
        # data variable.
        if dvar is None:
            context_means = np.repeat(context_means, self._get_n_data_variables(source_name))
            context_stds = np.repeat(context_stds, self._get_n_data_variables(source_name))

        normalized_context = (context - torch.tensor(context_means)) / torch.tensor(context_stds)

        # Values normalization
        if dvar is None:
            data_means = np.array(
                [self.data_means[source_name][var] for var in source.data_vars]
            )  # (C,)
            data_stds = np.array(
                [self.data_stds[source_name][var] for var in source.data_vars]
            )  # (C,)
        else:
            data_means = self.data_means[source_name][dvar]  # scalar value
            data_stds = self.data_stds[source_name][dvar]  # scalar value
        normalized_values = (values - torch.tensor(data_means)) / torch.tensor(data_stds)
        return normalized_context, normalized_values

    def __len__(self):
        return len(self.df)

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
            source.name: source.n_context_variables()
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
