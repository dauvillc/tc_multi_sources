"""Implements the MultiSourceDataset class."""

import torch
import pandas as pd
from multi_sources.data_processing.single_2d_source_dataset import Single2DSourceDataset


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.

    The dataset yields maps {source_name: None or (S, DT, C, D, V)} where:
    - S is the source index.
    - DT is the time delta between the reference time and the time of the element, in hours.
    - C is a tensor of shape (2, H, W) containing the coordinates (lat, lon) of each pixel.
    - D is a tensor of shape (H, W) containing the distance at each pixel to the center of the storm.
    - V is a tensor of shape (channels, H, W) containing the values of each pixel.
    The selection of the element returned for each source is done as follows (for an index i):
    - Select the ith element, all sources and all storms included.
    - Let t0 be the time of that element:
      - For each source, select the element with the closest time to t0 that is before t0.
        If there is no such element, return None for that source instead.
        If there is such an element but (t0 - t) > dt_max, return None for that source instead.
        Otherwise, return the element.
    """

    def __init__(self, sources, dt_max=24):
        """
        Args:
            sources (list of :obj:`multi_source.data_processing.Source`): The sources to use.
            dt_max (int): The maximum time delta between the elements returned for each source,
                in hours.
        """
        self.sources = sources
        self.dt_max = pd.Timedelta(dt_max, unit="h")
        # Create a Single2DSourceDataset for each source
        self.datasets = [Single2DSourceDataset(source) for source in sources]
        # Create a DataFrame gathering all of these coordinates as well as the index of the source.
        # The DF should have the columns 'SID', 'source', 'season', 'basin', 'cyclone_number', 'time'.
        self.source_dfs = [
            ds.data[["SID", "time", "season", "basin", "cyclone_number"]]
            .reset_index("sample")
            .to_dataframe()
            .assign(source=[i] * len(ds))
            for i, ds in enumerate(self.datasets)
        ]
        self.df = pd.concat(self.source_dfs)
        # Sort by 'SID', 'source', and 'time'
        self.source_dfs = [df.sort_values(["SID", "time"]) for df in self.source_dfs]
        self.df = self.df.sort_values(["SID", "source", "time"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve the storm/time pair corresponding to that index
        t0 = self.df.loc[idx]["time"]
        sid = self.df.loc[idx]["SID"]
        # Initialize the output dictionary
        output = {}
        # For each source, find the element with the closest time to t0 that is before t0
        for source_idx, df in enumerate(self.source_dfs):
            # Identify the rows corresponding to the storm
            df_storm = df[df["SID"] == sid]
            # Keep only the rows that are before t0
            df_storm = df_storm[df_storm["time"] <= t0]
            # If there is no such element, return None for that source
            if len(df_storm) == 0:
                output[self.sources[source_idx].name] = None
                continue
            # Find the closest time to t0
            t = df_storm["time"].iloc[-1]
            dt = t0 - t
            # If dt > dt_max, return None for that source
            if dt > self.dt_max:
                output[self.sources[source_idx].name] = None
                continue
            # Retrieve the element from the single source dataset
            C, D, V = self.datasets[source_idx].get_sample(sid, t)
            output[self.sources[source_idx].name] = (
                source_idx,
                dt.total_seconds() / 3600,
                torch.tensor(C, dtype=torch.float32),
                torch.tensor(D, dtype=torch.float32),
                torch.tensor(V, dtype=torch.float32),
            )

        return output
