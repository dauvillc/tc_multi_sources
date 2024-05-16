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

    Each element is associated with a storm and a synoptic time. For a given storm S and time t0, the dataset
    returns the element from each source that is closest to t0 and between t0 and t0 - dt_max.
    """

    def __init__(
        self,
        sources,
        include_seasons=None,
        exclude_seasons=None,
        dt_max=24,
        min_available_sources_prop=0.6,
    ):
        """
        Args:
            sources (list of :obj:`multi_source.data_processing.Source`): The sources to use.
            include_seasons (list of int): The years to include in the dataset. If None, all years are included.
            exclude_seasons (list of int): The years to exclude from the dataset. If None, no years are excluded.
            dt_max (int): The maximum time delta between the elements returned for each source,
                in hours.
            min_available_sources_prop (float): For a given sample (storm/time pair), the minimum proportion of
                sources that must have an available element for the sample to be included in the dataset.
        """
        self.sources = sources
        self.dt_max = pd.Timedelta(dt_max, unit="h")
        # Create a Single2DSourceDataset for each source
        self.datasets = [Single2DSourceDataset(source) for source in sources]
        # Create a DataFrame gathering all of these coordinates as well as the index of the source.
        # The DF should have the columns 'SID', 'source', 'season', 'basin', 'cyclone_number', 'time'.
        self.df = (
            pd.concat(
                [
                    ds.data[["SID", "time", "season", "basin", "cyclone_number"]]
                    .reset_index("sample")
                    .to_dataframe()
                    .assign(source=[i] * len(ds))
                    .assign(source_name=[ds.source.name] * len(ds))
                    for i, ds in enumerate(self.datasets)
                ]
            )
            .reset_index(drop=True)
            .sort_values(["SID", "source", "time"])
        )
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
            self.df.groupby(["SID", "syn_time"])["source"]
            .nunique()
            .rename("avail_source")
            .reset_index()
        )
        avail["avail_frac"] = avail["avail_source"] / self.get_n_sources()
        # - Keep only the storm/time pairs for which at least min_available_sources_prop sources are available
        avail = avail[avail["avail_frac"] >= min_available_sources_prop]
        self.df = self.df.merge(avail[["SID", "syn_time"]], on=["SID", "syn_time"], how="inner")
        # - Sort by SID,source,time and for every (SID,syn_time,source) triplet, keep only the last one
        #   (i.e. the one with the latest time)
        self.df = self.df.sort_values(["SID", "source", "time"]).drop_duplicates(
            ["SID", "syn_time", "source"], keep="last"
        )
        # Finally, compute the list of unique (SID,syn_time) pairs. Each row of this final dataframe will
        # constitute a sample of the dataset, while self.df can be used to retrieve the corresponding elements.
        self.samples_df = self.df[["SID", "syn_time"]].drop_duplicates().reset_index(drop=True)

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        # Retrieve the storm/time pair corresponding to that index
        sid = self.samples_df["SID"].iloc[idx]
        t0 = self.samples_df["syn_time"].iloc[idx]
        # Retrieve all elements from all sources that correspond to that storm/time pair
        sample = self.df[(self.df["SID"] == sid) & (self.df["syn_time"] == t0)]
        # Initialize the output dictionary
        output = {}
        # For each source, find the element with the closest time to t0 that is before t0
        for source_idx in range(self.get_n_sources()):
            source_idx_tensor = torch.tensor(source_idx, dtype=torch.float32)
            # Isolate the elements from that source
            sample_source = sample[sample["source"] == source_idx]
            # If there is no element for that source, fill with NaNs
            if len(sample_source) == 0:
                # Retrieve the height and width of the source
                h, w = self.datasets[source_idx].get_spatial_size()
                channels = self.datasets[source_idx].get_n_variables()
                # Fill with NaNs
                output[self.sources[source_idx].name] = (
                    source_idx_tensor,
                    torch.full((1,), float("nan")),
                    torch.full((2, h, w), float("nan")),
                    torch.full((h, w), float("nan")),
                    torch.full((channels, h, w), float("nan")),
                )
            else:
                # Retrieve the element from the single source dataset
                c, d, v = self.datasets[source_idx].get_sample(sid, sample_source["time"].iloc[0])
                output[self.sources[source_idx].name] = (
                    source_idx_tensor,
                    torch.tensor(
                        (t0 - sample_source["time"].iloc[0]).total_seconds() / 3600,
                        dtype=torch.float32,
                    ),
                    c,
                    d,
                    v,
                )

        return output

    def get_n_variables(self):
        """Returns a dict {source_name: n_variables}."""
        return {source.name: source.n_variables() for source in self.sources}

    def get_n_sources(self):
        """Returns the number of sources."""
        return len(self.sources)
