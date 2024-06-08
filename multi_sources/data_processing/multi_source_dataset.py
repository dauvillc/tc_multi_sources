"""Implements the MultiSourceDataset class."""

import torch
import pandas as pd
import numpy as np
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
from pathlib import Path


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.
    A sample from the dataset is a dict {source_name: (A, S, DT, C, D, V)}, where:
    - A is a scalar tensor of shape (1,) containing 1 if the element is available and -1 otherwise.
    - S is a scalar tensor of shape (1,) containing the index of the source.
    - DT is a scalar tensor of shape (1,) containing the time delta between the synoptic time
      and the element's time, normalized by dt_max.
    - C is a tensor of shape (3, H, W) containing the latitude, longitude, and land-sea mask.
    - D is a tensor of shape (H, W) containing the distance to the center of the storm.
    - V is a tensor of shape (n_variables, H, W) containing the variables for the source.

    Each element is associated with a storm and a synoptic time. For a given storm S and time t0,
    the dataset returns the element from each source that is closest
    to t0 and between t0 and t0 - dt_max.
    If for a given source, no element is available for the storm S and time t0, the element is
    filled with NaNs.
    """

    def __init__(
        self,
        metadata_path,
        dataset_dir,
        sources,
        include_seasons=None,
        exclude_seasons=None,
        dt_max=24,
        min_available_sources_prop=0.6,
        enable_data_augmentation=False,
    ):
        """
        Args:
            metadata_path (str): The path to the metadata file. The metadata file should be
                a CSV file with the
                columns 'sid', 'source_name', 'season', 'basin', 'cyclone_number', 'time'.
            dataset_dir (str): The directory containing the preprocessed dataset.
            sources (list of :obj:`Source`): The sources to include in the dataset.
            include_seasons (list of int): The years to include in the dataset.
                If None, all years are included.
            exclude_seasons (list of int): The years to exclude from the dataset.
                If None, no years are excluded.
            dt_max (int): The maximum time delta between the elements returned for each source,
                in hours.
            min_available_sources_prop (float): For a given sample (storm/time pair),
                the minimum proportion of sources that must have an available element
                for the sample to be included in the dataset.
            enable_data_augmentation (bool): If True, data augmentation is enabled.
        """
        self.dt_max = pd.Timedelta(dt_max, unit="h")
        self.dataset_dir = Path(dataset_dir)
        self.sources = sources
        self.enable_data_augmentation = enable_data_augmentation
        # Load the metadata file
        self.df = pd.read_json(metadata_path, orient="records", lines=True, convert_dates=["time"])
        # Filter the dataframe to only keep the rows where source_name is the name of a source
        # in self.sources
        source_names = [source.name for source in self.sources]
        self.source_variables = {source.name: source.variables for source in self.sources}
        self.df = self.df[self.df["source_name"].isin(source_names)]
        # Associate each source name with an index
        self.source_names = self.df["source_name"].unique()
        self.source_indices = {
            source_name: idx for idx, source_name in enumerate(self.source_names)
        }
        self.df["source"] = self.df["source_name"].map(self.source_indices)
        # We'll need to know the shape of the variables to create the tensors.
        self.source_shapes = {}
        for source_name in self.source_names:
            # Load a single file from that source to get the shape of the variables
            source_sample = self.df[self.df["source_name"] == source_name].iloc[0]
            season, basin, sid, time = (
                source_sample["season"],
                source_sample["basin"],
                source_sample["sid"],
                source_sample["time"],
            )
            arr = np.load(self.get_data_filepath(season, basin, sid, time, source_name))
            self.source_shapes[source_name] = arr.shape[1:]  # Remove the channel dimension
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
        # (sid, syn_time) pairs that will be used to sample the dataset.
        # Some more processing is apply to it, which was first written in the notebook
        # notebooks/sources.ipynb.
        # Compute the synoptic time that is closest and after the element's time
        self.df["syn_time"] = self.df["time"].dt.ceil("6h")
        # For each element, compute the list of synoptic times that are within dt_max
        # of the element's time, e.g.
        # 06:41:00 -> [12:00:00, 18:00:00, 00:00:00 (next day), 06:00:00 (next day)]
        self.df["syn_time"] = self.df.apply(
            lambda row: pd.date_range(row["syn_time"], row["time"] + self.dt_max, freq="6h"),
            axis=1,
        )
        # Explode the synoptic times into separate rows
        self.df = self.df.explode("syn_time")
        # Filtering:
        # - Compute for each storm/time the number of available sources
        avail = (
            self.df.groupby(["sid", "syn_time"])["source"]
            .nunique()
            .rename("avail_source")
            .reset_index()
        )
        avail["avail_frac"] = avail["avail_source"] / self.get_n_sources()
        # - Keep only the storm/time pairs for which at least min_available_sources_prop sources
        # are available
        avail = avail[avail["avail_frac"] >= min_available_sources_prop]
        self.df = self.df.merge(avail[["sid", "syn_time"]], on=["sid", "syn_time"], how="inner")
        # - Sort by sid,source,time and for every (sid,syn_time,source) triplet,
        # keep only the last one (i.e. the one with the latest time)
        self.df = self.df.sort_values(["sid", "source", "time"]).drop_duplicates(
            ["sid", "syn_time", "source"], keep="last"
        )
        # Finally, compute the list of unique (sid,syn_time) pairs.
        # Each row of this final dataframe will
        # constitute a sample of the dataset, while self.df can be used
        # to retrieve the corresponding elements.
        self.samples_df = self.df[["sid", "syn_time"]].drop_duplicates().reset_index(drop=True)

        # Data augmentation: prepare the transformations
        if self.enable_data_augmentation:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(180),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                ]
            )

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
        syn_time = sample["syn_time"]
        output = {}
        # Isolate the rows of self.df corresponding to the given sid and syn_time
        sample_df = self.df[(self.df["sid"] == sid) & (self.df["syn_time"] == syn_time)]
        season, basin = sample_df["season"].iloc[0], sample_df["basin"].iloc[0]
        # For each source, try to load the element at the given time
        for source_name in self.source_names:
            source_tensor = torch.tensor(self.source_indices[source_name], dtype=torch.float32)
            # Try to find an element for the given source, sid and syn_time
            df = sample_df[sample_df["source_name"] == source_name]
            if len(df) == 0 or syn_time - df["time"].iloc[0] > self.dt_max:
                # No element available for this source
                # Create a tensor of the appropriate shape filled with NaNs
                avail_tensor = torch.tensor(-1, dtype=torch.float32)
                source_shape = self.source_shapes[source_name]
                source_channels = self.get_n_variables()[source_name]
                output[source_name] = (
                    avail_tensor,  # A
                    source_tensor,  # S
                    torch.tensor(float("nan"), dtype=torch.float32),  # DT
                    torch.full(
                        (3, *source_shape), fill_value=float("nan"), dtype=torch.float32
                    ),  # C
                    torch.full(source_shape, fill_value=float("nan"), dtype=torch.float32),  # D
                    torch.full(
                        (source_channels, *source_shape),
                        fill_value=float("nan"),
                        dtype=torch.float32,
                    ),  # V
                )
            else:
                time = df["time"].iloc[0]
                dt = time - syn_time
                avail_tensor = torch.tensor(1, dtype=torch.float32)
                dt_tensor = torch.tensor(
                    dt.total_seconds() / self.dt_max.total_seconds(), dtype=torch.float32
                )
                # Load the npy file containing the data for the given storm, time, and source
                filepath = self.get_data_filepath(season, basin, sid, time, source_name)
                tensor = torch.from_numpy(np.load(filepath)).to(torch.float32)
                # The tensor has shape (channels, H, W), where the channels are in the following
                # order:
                # - Latitude
                # - Longitude
                # - Land-sea mask
                # - Distance to the center of the storm
                # - Variables
                # Create the coordinate tensor
                C = tensor[:3]
                # Create the distance tensor
                D = tensor[3]
                # For the variables, we need to first retrieve which variables to use from
                # the config. We can access these as source.variables.
                # To know the order of the available variables in the tensor, we can use
                # df['data_vars'], which is the list [var1, ..., varN] of the variables
                # in the tensor.
                selected_vars = self.source_variables[source_name]
                data_vars = df["data_vars"].iloc[0]
                var_indices = [4 + data_vars.index(var) for var in selected_vars]
                V = tensor[var_indices]
                # Data augmentation
                if self.enable_data_augmentation:
                    # Convert C, D, and V to tv_tensors.Image so that they're all transformed
                    # with the same transformation
                    C, V = tv_tensors.Image(C), tv_tensors.Image(V)
                    D = tv_tensors.Image(D.unsqueeze(0))
                    # Apply the same transformation to C, D, and V
                    C, D, V = self.transform(C, D, V)
                    D = D.squeeze(0)
                output[source_name] = (avail_tensor, source_tensor, dt_tensor, C, D, V)

        return output

    def get_data_filepath(self, season, basin, sid, time, source_name):
        """Returns the path to the file containing the data for the given storm, time, and source.

        Args:
            season (int or str): The season of the storm.
            basin (str): The basin of the storm.
            sid (str): The storm ID.
            time (pd.Timestamp): The time of the element.
            source_name (str): The name of the source.

        Returns:
            Path: The path to the file containing the data.
        """
        path = self.dataset_dir / f"{season}/{basin}/{sid}"
        return path / f"{time.strftime('%Y%m%d%H%M%S')}-{source_name}.npy"

    def __len__(self):
        return len(self.samples_df)

    def get_source_variables(self):
        """Returns a dict {source_name: [variable_name]}."""
        return {source.name: source.variables for source in self.sources}

    def get_n_variables(self):
        """Returns a dict {source_name: n_variables}."""
        return {source.name: len(source.variables) for source in self.sources}

    def get_n_sources(self):
        """Returns the number of sources."""
        return len(self.source_names)

    def get_source_sizes(self):
        """Returns a dict {source_name: (H, W)}."""
        return self.source_shapes
