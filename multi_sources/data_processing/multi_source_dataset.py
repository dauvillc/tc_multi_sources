"""Implements the MultiSourceDataset class."""

import torch
import pandas as pd
from multi_sources.data_processing.single_2d_source_dataset import Single2DSourceDataset


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.

    The dataset yields maps {source_name: (S, DT, C, D, V)} where:
    - S is the source index.
    - DT is the time delta between the reference time and the time of the element, in hours.
    - C is a tensor of shape (2, H, W) containing the coordinates (lat, lon) of each pixel.
    - D is a tensor of shape (H, W) containing the distance at each pixel to the center of the storm.
    - V is a tensor of shape (channels, H, W) containing the values of each pixel.
    If the element is not available for a source, DT, C, D, and V are filled with NaNs.
    The selection of the element returned for each source is done as follows (for an index i):
    - Select the ith element, all sources and all storms included.
    - Let t0 be the time of that element:
      - For each source, select the element with the closest time to t0 that is before t0.
        If there is no such element, return None for that source instead.
        If there is such an element but (t0 - t) > dt_max, return None for that source instead.
        Otherwise, return the element.
    """

    def __init__(self, sources, include_seasons=None, exclude_seasons=None, dt_max=24):
        """
        Args:
            sources (list of :obj:`multi_source.data_processing.Source`): The sources to use.
            include_seasons (list of int): The years to include in the dataset. If None, all years are included.
            exclude_seasons (list of int): The years to exclude from the dataset. If None, no years are excluded.
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
        # Check that the included and excluded seasons are not overlapping
        if include_seasons is not None and exclude_seasons is not None:
            if set(include_seasons).intersection(exclude_seasons):
                raise ValueError("The included and excluded seasons are overlapping.")
        # Select the seasons to include
        if include_seasons is not None:
            self.source_dfs = [df[df["season"].isin(include_seasons)] for df in self.source_dfs]
        # Select the seasons to exclude
        if exclude_seasons is not None:
            self.source_dfs = [df[~df["season"].isin(exclude_seasons)] for df in self.source_dfs]
        self.df = pd.concat(self.source_dfs).reset_index(drop=True)
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
            source_idx_tensor = torch.tensor(source_idx, dtype=torch.float32)
            # Identify the rows corresponding to the storm
            df_storm = df[df["SID"] == sid]
            # Keep only the rows that are before t0
            df_storm = df_storm[df_storm["time"] <= t0]
            # If there is no such element, return None for that source
            if len(df_storm) == 0 or (t0 - df_storm["time"].iloc[-1]) > self.dt_max:
                # Retrieve the height and width of the source from the single source dataset
                H, W = self.datasets[source_idx].get_spatial_size()
                n_channels = self.datasets[source_idx].get_n_variables()
                output[self.sources[source_idx].name] = (
                    source_idx_tensor,
                    torch.tensor(0.0, dtype=torch.float32),
                    torch.full((2, H, W), float("nan"), dtype=torch.float32),
                    torch.full((H, W), float("nan"), dtype=torch.float32),
                    torch.full((n_channels, H, W), float("nan"), dtype=torch.float32),
                )
                continue
            # Find the closest time to t0
            t = df_storm["time"].iloc[-1]
            dt = t0 - t
            # Retrieve the element from the single source dataset
            C, D, V = self.datasets[source_idx].get_sample(sid, t)
            output[self.sources[source_idx].name] = (
                source_idx_tensor,
                torch.tensor(dt.total_seconds() / 3600.0, dtype=torch.float32),
                torch.tensor(C, dtype=torch.float32),
                torch.tensor(D, dtype=torch.float32),
                torch.tensor(V, dtype=torch.float32),
            )

        return output

    def get_n_variables(self):
        """Returns a dict {source_name: n_variables}."""
        return {source.name: source.n_variables() for source in self.sources}
