"""Implements the MultiSourceDataset class."""

import xarray as xr
import torch
import pandas as pd
import netCDF4 as nc
from pathlib import Path


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.

    The dataset yields maps {source_name: (S, DT, C, D, V)} where:
    - S is the source index.
    - DT is the time delta between the reference time and the time of the element, in hours.
    - C is a tensor of shape (2, H, W) containing the coordinates (lat, lon) of each pixel.
    - D is a tensor of shape (H, W) containing the distance at each pixel to the center of the storm.
    - V is a tensor of shape (channels, H, W) containing the values of each pixel.
    If the element is not available for a source, DT, C, D, and V are filled with NaNs.

    Each element is associated with a storm and a synoptic time. For a given storm S and time t0, the dataset
    returns the element from each source that is closest to t0 and between t0 and t0 - dt_max.
    """

    def __init__(
        self,
        metadata_path,
        dataset_dir,
        include_seasons=None,
        exclude_seasons=None,
        dt_max=24,
        min_available_sources_prop=0.6,
    ):
        """
        Args:
            metadata_path (str): The path to the metadata file. The metadata file should be a CSV file with the
                columns 'sid', 'source_name', 'season', 'basin', 'cyclone_number', 'time'.
            dataset_dir (str): The directory containing the preprocessed dataset. The structure should be
                dataset_dir/season/basin/sid.nc.
            include_seasons (list of int): The years to include in the dataset. If None, all years are included.
            exclude_seasons (list of int): The years to exclude from the dataset. If None, no years are excluded.
            dt_max (int): The maximum time delta between the elements returned for each source,
                in hours.
            min_available_sources_prop (float): For a given sample (storm/time pair), the minimum proportion of
                sources that must have an available element for the sample to be included in the dataset.
        """
        self.dt_max = pd.Timedelta(dt_max, unit="h")
        self.dataset_dir = Path(dataset_dir)
        # Load the metadata file
        self.df = pd.read_csv(metadata_path, parse_dates=["time"])
        # Associate each source name with an index
        self.source_names = self.df["source_name"].unique()
        self.source_indices = {
            source_name: idx for idx, source_name in enumerate(self.source_names)
        }
        self.df["source"] = self.df["source_name"].map(self.source_indices)
        # We'll need to know the names of the variables in each source to access them and
        # to know the number of channels in the tensors.
        self.source_variables = {}
        for source_name in self.source_names:
            # Isolate the first element for each source
            first_element = self.df[self.df["source_name"] == source_name].iloc[0]
            # Open the file and count the number of variables
            # The files are stored under dataset_dir/season/basin/sid.nc
            with xr.open_dataset(
                self.dataset_dir
                / str(first_element["season"])
                / first_element["basin"]
                / f"{first_element['sid']}.nc",
                group=source_name,
            ) as sample:
                # Only include the variables that are not coordinates and are not
                # 'dist_to_center', 'latitude', 'longitude'.
                self.source_variables[source_name] = [
                    var
                    for var in sample.data_vars
                    if var not in ["dist_to_center", "latitude", "longitude"]
                ]
        # Create a DataFrame gathering all of these coordinates as well as the index of the source.
        # The DF should have the columns 'sid', 'source', 'season', 'basin', 'cyclone_number', 'time'.
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
        # Compute the synoptic time that is closest and after the element's time
        self.df["syn_time"] = self.df["time"].dt.ceil("6h")
        # For each element, compute the list of synoptic times that are within dt_max of the element's time
        # e.g. 06:41:00 -> [12:00:00, 18:00:00, 00:00:00 (next day), 06:00:00 (next day)]
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
        # - Keep only the storm/time pairs for which at least min_available_sources_prop sources are available
        avail = avail[avail["avail_frac"] >= min_available_sources_prop]
        self.df = self.df.merge(avail[["sid", "syn_time"]], on=["sid", "syn_time"], how="inner")
        # - Sort by sid,source,time and for every (sid,syn_time,source) triplet, keep only the last one
        #   (i.e. the one with the latest time)
        self.df = self.df.sort_values(["sid", "source", "time"]).drop_duplicates(
            ["sid", "syn_time", "source"], keep="last"
        )
        # Finally, compute the list of unique (sid,syn_time) pairs. Each row of this final dataframe will
        # constitute a sample of the dataset, while self.df can be used to retrieve the corresponding elements.
        self.samples_df = self.df[["sid", "syn_time"]].drop_duplicates().reset_index(drop=True)

    def __getitem__(self, idx):
        """Returns the element at the given index."""
        pass

    def __len__(self):
        return len(self.samples_df)

    def get_n_variables(self):
        """Returns a dict {source_name: n_variables}."""
        return {source_name: len(variables) for source_name, variables in self.source_variables.items()}

    def get_n_sources(self):
        """Returns the number of sources."""
        return len(self.source_names)
