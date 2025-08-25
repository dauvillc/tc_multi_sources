"""Implements the MultiSourceDataset class."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from tqdm import tqdm

from multi_sources.data_processing.grid_functions import crop_nan_border
from multi_sources.data_processing.source import Source
from multi_sources.data_processing.utils import (
    compute_sources_availability,
    load_nc_with_nan,
)


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.
    A sample from the dataset is a dict {(source_name, index): data} where
    source_name is the name of the source and index is an integer representing
    the observation index (0 = most recent, 1 = second most recent, etc.).
    Each data dict contains the following key-value pairs:
    - "avail" is a scalar tensor of shape (1,) containing 1 if the element is available
        and -1 otherwise.
    - "dt" is a scalar tensor of shape (1,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "characs" is a tensor of shape (n_charac_vars,) containing the source characteristics.
        Each data variable within a source has its own charac variables, which are all
        concatenated into a single tensor.
    - "coords" is a tensor of shape (2, H, W) containing the latitude and longitude at each pixel.
    - "landmask" is a tensor of shape (H, W) containing the land mask.
    - "dist_to_center" is a tensor of shape (H, W) containing the distance
        to the center of the storm.
    - "values" is a tensor of shape (n_variables, H, W) containing the variables for the source.

    For a given storm S and reference time t0, the dataset will yield observations from
    the sources that are available in the time window (t0 - dt_max, t0) for each source.
    Multiple observations from the same source can be available in that time window.
    """

    def __init__(
        self,
        dataset_dir,
        split,
        included_variables_dict,
        dt_max=24,
        min_available_sources=0,
        source_types_min_avail={},
        source_types_max_avail={},
        select_most_recent=False,
        min_ref_time_delta=0,
        num_workers=0,
        forecasting_lead_time=None,
        forecasting_sources=None,
        min_tc_intensity=None,
        mask_spatial_coords=[],
        data_augmentation=None,
        seed=42,
    ):
        """
        Args:
            dataset_dir (str): The directory containing the preprocessed dataset.
                The directory should contain the following:
                - train/val/test.json: pandas DataFrame containing the samples metadata.
                - processed/
                    - source_name/
                        - source_metadata.json: dictionary containing the metadata for the source.
                        - samples_metadata.json: dictionary containing the metadata for
                            the samples.
                        - *.nc: netCDF files containing the data for the source.
                - constants/
                    - source_name/
                        - data_means/stds.json: dictionaries containing the means and stds
                            for the data variables of the source.
                        - charac_vars_min_max.json: dictionaries {charac_var_name: {min, max}}
            split (str): The split of the dataset. Must be one of "train", "val", or "test".
            included_variables_dict (dict): A dictionary
                {source_name: (list of variables, list of input-only variables)}
                containing the variables (i.e. channels) to include for each source.
                Variables also included in the second list will be included in the yielded
                data but will be flagged as input-only in the Source object.
            dt_max (int): The maximum time delta between the elements returned for each source,
                in hours.
            min_available_sources (int): The minimum number of sources that must be available
                for a sample to be included in the dataset. Does NOT include the forecast source
                if it is enabled.
            source_types_min_avail (dict of str to int): A dictionary D = {src_type: availability}
                such that if D[source_type] is the minimum number of sources of type source_type
                that must be available for a sample to be included in the dataset.
            source_types_max_avail (dict of str to int): A dictionary D = {src_type: availability}
                such that for a source type T, at most D[T] sources of that type can be included
                in a sample.
            select_most_recent (bool): If True, prioritize the most recent observations within
                a source when there are too many available sources of that type. Otherwise,
                prioritize randomly.
            min_ref_time_delta (int): The minimum time delta between two reference times
                of the same storm (ie between two samples of the same storm).
            num_workers (int): If > 1, number of workers to use for parallel loading of the data.
            forecasting_lead_time (int): If not None, the lead time to use for forecasting.
                Must be used in conjunction with forecasting_sources. If enabled, the reference
                source will always be one of the forecast source.
                The time window for the other sources
                will be (t0 - lead_time - dt_max, t0 - lead_time), i.e. the other sources will be
                in a window of duration dt_max, ending at t0 - lead_time.
            forecasting_sources (list of str): If not None, the name of the sources to use
                for forecasting.
                Must be used in conjunction with forecasting_lead_time.
            min_tc_intensity (float): If not None, will filter the reference samples to only keep
                the ones where the TC intensity is above the given threshold at the reference time.
            mask_spatial_coords (list of str): If not None, names of sources whose spatial
                coordinates will be masked (set to zeros).
            data_augmentation (None or MultiSourceDataAugmentation): If not None, instance
                of MultiSourceDataAugmentation to apply to the data.
            seed (int): The seed to use for the random number generator.
        """

        self.dataset_dir = Path(dataset_dir)
        self.constants_dir = self.dataset_dir / "constants"
        self.processed_dir = self.dataset_dir / "prepared"
        self.split = split
        self.dt_max = pd.Timedelta(dt_max, unit="h")
        self.source_types_max_avail = source_types_max_avail
        self.min_ref_time_delta = min_ref_time_delta
        self.select_most_recent = select_most_recent
        self.mask_spatial_coords = mask_spatial_coords
        self.data_augmentation = data_augmentation
        self.rng = np.random.default_rng(seed)

        print(f"{split}: Browsing requested sources and loading metadata...")
        # Load and merge individual source metadata files
        sources_metadata = {}
        for source_name in included_variables_dict:
            source_metadata_path = self.processed_dir / source_name / "source_metadata.json"
            # If the source metadata file is not found, print a warning
            # and skip the source
            if not source_metadata_path.exists():
                print(f"Warning: {source_metadata_path} not found. Skipping source {source_name}.")
                continue
            with open(source_metadata_path, "r") as f:
                sources_metadata[source_name] = json.load(f)
        # If no source has been found, raise an error
        if len(sources_metadata) == 0:
            raise ValueError("Did not find any source metadata files.")
        # Filter the included variables for each source
        self.variables_dict = {
            source: included_variables_dict[source] for source in sources_metadata
        }

        self.sources, self.sources_dict = [], {}
        for source_name, (all_vars, input_only, output_only) in self.variables_dict.items():
            # Update the source metadata with the included variables
            sources_metadata[source_name]["data_vars"] = all_vars
            sources_metadata[source_name]["input_only_vars"] = input_only
            sources_metadata[source_name]["output_only_vars"] = output_only
            # Create the source object
            self.sources.append(Source(**sources_metadata[source_name]))
            self.sources_dict[source_name] = self.sources[-1]
        # Load the samples metadata based on the split
        self.df = pd.read_csv(self.dataset_dir / f"{split}.csv", parse_dates=["time"])

        # Filter the dataframe to only keep the rows where source_name is the name of a source
        # in self.sources
        source_names = [source.name for source in self.sources]
        self.source_variables = {source.name: source.data_vars for source in self.sources}
        self.source_types = list(set([source.type for source in self.sources]))
        self.df = self.df[self.df["source_name"].isin(source_names)]
        self.df = self.df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No samples found for the given sources.")

        if forecasting_lead_time is not None:
            if forecasting_sources is None:
                raise ValueError("forecasting_lead_time must be used with forecasting_sources.")
            if not isinstance(forecasting_sources, list):
                forecasting_sources = [forecasting_sources]
            self.forecasting_lead_time = pd.Timedelta(forecasting_lead_time, unit="h")
            self.forecasting_sources = [s for s in forecasting_sources if s in source_names]
            print("Forecasting sources: ", self.forecasting_sources)
            print("Forecasting lead time: ", self.forecasting_lead_time)
        else:
            self.forecasting_lead_time = pd.Timedelta(0, unit="h")
            self.forecasting_sources = None

        # ========================================================================================
        # BUILDING THE REFERENCE DATAFRAME
        # We'll now filter the samples to only keep the ones where at least min_available_sources
        # sources are available.
        # - We'll compute a dataframe D of shape (n_samples, n_sources) such that
        # D[i, s] = 1 if source i is available for sample s, and 0 otherwise.
        print(f"{split}: Computing sources availability...")
        available_sources = compute_sources_availability(
            self.df, self.dt_max, lead_time=self.forecasting_lead_time, num_workers=num_workers
        )
        # We can now build the reference dataframe:
        # - self.df contains the metadata for all elements from all sources,
        # - self.reference_df is a subset of self.df containing every (sid, time) pair
        #   that defines a sample which matches the availability criteria.
        # There are two such criteria to check: the overall number of available sources
        # and the number of available sources of each type.
        # The criteria will fill a list of boolean masks, which will be combined
        # with a logical AND to filter the dataframe.
        masks = []
        # First: only keep the samples where at least min_available_sources sources are available
        available_sources_count = available_sources.sum(axis=1)
        masks.append(available_sources_count >= min_available_sources)
        self.min_available_sources = min_available_sources
        # Second: check the availability of the sources of each type specifically.
        # If a source type is in the source types of self.sources but was not specified in the
        # source_types_min_avail dict, we'll interpret it as a minimum availability of 0.
        self.source_types_min_avail = {
            source_type: source_types_min_avail.get(source_type, 0)
            for source_type in self.source_types
        }
        # For each source type, compute the number of available sources in each sample
        source_types_availability = {}  # {source_type: n available sources of this type}
        for source_type, required_avail in self.source_types_min_avail.items():
            # Get the sources of the given type
            sources_of_type = [
                source.name
                for source in self.sources
                if source.type == source_type and source.name in available_sources.columns
            ]
            # Compute the number of available sources of this type
            st_avail_count = available_sources[sources_of_type].sum(axis=1)
            # Compute the mask for the samples that have at least required_avail sources
            masks.append(st_avail_count >= required_avail)
            source_types_availability[source_type] = st_avail_count
        # Combine the masks with a logical AND
        mask = np.logical_and.reduce(masks)
        # We can now build the reference dataframe:
        self.reference_df = self.df[mask][["sid", "time", "source_name", "intensity"]]
        self.reference_df["n_available_sources"] = available_sources_count[mask]
        for source_type, st_avail_count in source_types_availability.items():
            self.reference_df["n_available_sources_" + source_type] = st_avail_count[mask]

        # If forecasting, only use the forecast sources as references
        if self.forecasting_sources is not None:
            self.reference_df = self.reference_df[
                self.reference_df["source_name"].isin(self.forecasting_sources)
            ]

        # Avoid duplicated samples if two observations are at the exact same time
        self.reference_df = self.reference_df.drop_duplicates(["sid", "time"])
        self.reference_df = self.reference_df.reset_index(drop=True)

        # Optional intensity filtering
        if min_tc_intensity is not None:
            self.reference_df = self.reference_df[
                self.reference_df["intensity"] >= min_tc_intensity
            ]

        # If required, sample the reference dataframe to keep only one sample
        # every min_ref_time_delta hours.
        if self.min_ref_time_delta > 0:
            # Assign each row to a time bin
            self.reference_df["time_bin"] = self.reference_df["time"].dt.floor(
                f"{self.min_ref_time_delta}H"
            )
            # Sort to ensure the first occurrence is the earliest in each bin
            self.reference_df = self.reference_df.sort_values(["sid", "time"])
            # Drop duplicates to keep the first occurrence in each bin per sid
            self.reference_df = self.reference_df.drop_duplicates(subset=["sid", "time_bin"])
            # Drop the 'time_bin' column as it's no longer needed
            self.reference_df = self.reference_df.drop(columns="time_bin").reset_index(drop=True)

        # If no samples are left, raise an error
        if len(self.reference_df) == 0:
            raise ValueError(
                f"No samples left after filtering for {split} with min_available_sources="
                f"{min_available_sources} and min_ref_time_delta={self.min_ref_time_delta}."
            )

        # ========================================================================================
        # Pre-computation to speed up the data loading: instead of isolating the rows
        # of self.df in __getitem__, we'll pre-compute them now.
        # We'll obtain a list [sample_df_1, sample_df_2, ...] where each element corresponds to a
        # a row of self.reference_df (i.e. an item in self.__getitem__).
        print(f"{split}: Pre-computing the samples...")
        self.sample_dfs = []
        for _, row in tqdm(self.reference_df.iterrows(), total=len(self.reference_df)):
            sid, t0 = row["sid"], row["time"]
            sample_df = self.df[self.df["sid"] == sid]
            # Only keep the rows where the time is within the time window
            # defined by the reference time t0.
            min_t = t0 - self.forecasting_lead_time - self.dt_max
            max_t = t0 - self.forecasting_lead_time
            time_mask = (sample_df["time"] <= max_t) & (sample_df["time"] > min_t)
            # For the forecast sources, we also keep the rows where the time is
            # exactly the forecast time.
            if self.forecasting_sources:
                forecast_mask = (
                    sample_df["source_name"].isin(self.forecasting_sources) & sample_df["time"]
                    == t0
                )
                time_mask = time_mask | forecast_mask
            sample_df = sample_df[time_mask]

            # If we're selecting the sources chronologically, we can pre-compute the sorted order.
            if self.select_most_recent:
                sample_df = sample_df.sort_values("time", ascending=False)

            self.sample_dfs.append(sample_df)

        # ========================================================================================
        # Load the data means and stds
        self.data_means, self.data_stds = {}, {}
        for source in self.sources:
            # Data vars means and stds
            with open(self.constants_dir / source.name / "data_means.json", "r") as f:
                self.data_means[source.name] = json.load(f)
            with open(self.constants_dir / source.name / "data_stds.json", "r") as f:
                self.data_stds[source.name] = json.load(f)

        # Characteristic variables: pre-build the tensors of characteristics for each source,
        # since those do not depend on the sample.
        self.charac_vars_tensors = {}
        self.charac_vars_min, self.charac_vars_max = defaultdict(list), defaultdict(list)
        for source in self.sources:
            charac_vars = source.get_charac_values()
            self.charac_vars_tensors[source.name] = torch.tensor(charac_vars, dtype=torch.float32)
            # Also pre-load the min and max of the charac variables for the source. We'll fill
            # the self.charac_vars_min and self.charac_vars_max lists with the min and max
            # of the source's charac variables, in the same order as in the tensors we just built;
            # this way we can easily normalize the charac variables.
            with open(self.constants_dir / source.name / "charac_vars_min_max.json", "r") as f:
                source_characs_min_max = json.load(f)
            # iter_charac_variables() yields the items in the same order as get_charac_values().
            for charac_var_name, data_var_name, _ in source.iter_charac_variables():
                self.charac_vars_min[source.name].append(
                    source_characs_min_max[charac_var_name]["min"]
                )
                self.charac_vars_max[source.name].append(
                    source_characs_min_max[charac_var_name]["max"]
                )
            # convert to tensors
            self.charac_vars_min[source.name] = torch.tensor(
                self.charac_vars_min[source.name], dtype=torch.float32
            )
            self.charac_vars_max[source.name] = torch.tensor(
                self.charac_vars_max[source.name], dtype=torch.float32
            )

    def __getitem__(self, idx):
        """Returns the element at the given index.

        Args:
            idx (int): The index of the element to retrieve.

        Returns:
            sample (dict): A dictionary of the form {(source_name, index): data}
                where index is an integer representing the observation index
                (0 = most recent, 1 = second most recent, etc.)
        """
        sample_df = self.sample_dfs[idx]
        t0 = self.reference_df.iloc[idx]["time"]

        # If selecting the sources randomly, we do it here so that it changes
        # between epochs.
        if not self.select_most_recent:
            sample_df = sample_df.sample(frac=1, random_state=self.rng.integers(0, 1e6))

        # For each source type, only keep the maximum number of sources
        # indicate.
        sample_df = sample_df.groupby("source_type", group_keys=False).apply(
            lambda g: g.head(self.source_types_max_avail.get(g.name, len(g)))
        )

        # For each source, try to load the element at the given time
        output = {}
        sources_cnt = defaultdict(int)  # {source_name: number of obs from that source added}

        for _, row in sample_df.iterrows():
            source_name = row["source_name"]
            source = self.sources_dict[source_name]

            time = row["time"]
            dt = t0 - self.forecasting_lead_time - time
            DT = torch.tensor(
                dt.total_seconds() / self.dt_max.total_seconds(),
                dtype=torch.float32,
            )

            # Load the npy file containing the data for the given storm, time, and source
            filepath = Path(row["data_path"])
            with Dataset(filepath) as ds:
                # Coordinates
                lat = load_nc_with_nan(ds["latitude"])
                lon = load_nc_with_nan(ds["longitude"])
                # Make sure the longitude is in the range [-180, 180]
                lon = np.where(lon > 180, lon - 360, lon)
                C = np.stack([lat, lon], axis=0)
                C = torch.tensor(C, dtype=torch.float32)
                # Load the land mask and distance to the center of the storm
                LM = torch.tensor(load_nc_with_nan(ds["land_mask"]), dtype=torch.float32)
                D = torch.tensor(load_nc_with_nan(ds["dist_to_center"]), dtype=torch.float32)
                # If the spatial coordinates should be masked, set them to zeros
                if source_name in self.mask_spatial_coords:
                    C = torch.zeros_like(C)
                    LM = torch.zeros_like(LM)
                    D = torch.zeros_like(D)

                # Characterstic variables
                if source.n_charac_variables() == 0:
                    CA = None
                else:
                    CA = torch.tensor(source.get_charac_values(), dtype=torch.float32)

                # Values
                # Load the variables in the order specified in the source
                V = np.stack([load_nc_with_nan(ds[var]) for var in source.data_vars], axis=0)
                V = torch.tensor(V, dtype=torch.float32)

                # Normalize the characs and values tensors
                CA, V = self.normalize(V, source, characs=CA)

                # Assemble the output dict for that source
                output_source = {
                    "dt": DT,
                    "coords": C,
                    "landmask": LM,
                    "dist_to_center": D,
                    "values": V,
                }
                if CA is not None:
                    output_source["characs"] = CA

                # (Optional) Data augmentation
                if self.data_augmentation is not None:
                    output_source = self.data_augmentation(output_source, source.type)

                if source.dim == 2:
                    # For images only.
                    # The values can contain borders that are fully NaN (due to the sources
                    # originally having multiple channels that are not aligned geographically).
                    # Compute how much we can crop them, and apply that cropping to all spatial
                    # tensors to keep the spatial alignment.
                    V, C = output_source["values"], output_source["coords"]
                    LM, D = output_source["landmask"], output_source["dist_to_center"]
                    V, C, LM, D = crop_nan_border(V, [V, C, LM, D])
                    output_source["values"], output_source["coords"] = V, C
                    output_source["landmask"], output_source["dist_to_center"] = LM, D

                # Add the processed observation to the list for this source
                output[(source_name, sources_cnt[source_name])] = output_source
                sources_cnt[source_name] += 1

        if len(output) <= 1:
            sid, t0 = self.reference_df.iloc[idx][["sid", "time"]]
            print(sample_df)
            print(f"t0 = {t0}, sid = {sid}")
            raise ValueError(
                f"Sample {sid} at time {t0} has only {len(output)} sources available."
            )
        return output

    def normalize(
        self,
        values,
        source,
        characs=None,
        denormalize=False,
        leading_dims=0,
        device=None,
    ):
        """Normalizes the characs and values tensors associated with a given
        source, and optionally a specific data variable.
        Args:
            values (torch.Tensor): tensor of shape (C, ...) if leading_dims=0,
                or (D1, D2, ..., Dn, C, ...) for leading_dims=n, where ... are the spatial dimensions.
            source (Source or str): Source object representing the source, or name
                of the source.
            characs (torch.Tensor, optiona): tensor of shape (n_charac_vars,)
                containing the characteristic variables.
            denormalize (bool, optional): If True, denormalize the characs and values tensors
                instead of normalizing them.
            leading_dims (int, optional): Number of leading dimensions before the channel dimension.
                For example, if leading_dims=0, the tensor has shape (C, ...),
                if leading_dims=1, the tensor has shape (D1, C, ...),
                if leading_dims=2, the tensor has shape (D1, D2, C, ...), etc.
            device (torch.device, optional): Device to use for the normalization.
        Returns:
            normalized_characs (torch.Tensor): normalized charac variables.
            normalized_values (torch.Tensor): normalized values.
        """
        if isinstance(source, Source):
            source_name = source.name
        else:
            source_name = source
            source = self.sources_dict[source_name]

        # Characteristic variables normalization.
        if characs is not None:  # Values normalization.
            characs = characs.to(torch.float32)
            min, max = self.charac_vars_min[source_name], self.charac_vars_max[source_name]
            if device is not None:
                min = min.to(device)
                max = max.to(device)
            if denormalize:
                normalized_characs = characs * (max - min) + min
            else:
                normalized_characs = (characs - min) / (max - min)
        else:
            normalized_characs = None

        data_vars = source.data_vars

        # Create shape with leading 1s for each leading dimension, followed by channel dimension
        # and trailing 1s for spatial dimensions
        shape = [1] * leading_dims + [-1] + [1] * (len(values.shape) - leading_dims - 1)

        data_means = np.array([self.data_means[source_name][var] for var in data_vars]).reshape(
            shape
        )
        data_stds = np.array([self.data_stds[source_name][var] for var in data_vars]).reshape(
            shape
        )
        data_means = torch.tensor(data_means, dtype=values.dtype)
        data_stds = torch.tensor(data_stds, dtype=values.dtype)

        if device is not None:
            data_means = data_means.to(device)
            data_stds = data_stds.to(device)
        if denormalize:
            normalized_values = values * data_stds + data_means
        else:
            normalized_values = (
                values - data_means
            ) / data_stds  # Fixed: was dividing by data_means

        return normalized_characs, normalized_values

    def __len__(self):
        return len(self.reference_df)

    def _get_source_names(self):
        """Returns a list of the source names (before splitting the sources)."""
        return [source.name for source in self.sources]

    def _get_n_sources(self):
        """Returns the number of original sources (before splitting the sources)."""
        return len(self.sources)

    def _get_n_data_variables(self, source_name=None):
        """Returns either the number of data variables within a source,
        or a dict {source_name: number of data variables}."""
        if source_name is not None:
            return self._get_n_data_variables()[source_name]
        return {source.name: source.n_data_variables() for source in self.sources}

    def _get_n_charac_variables(self, source_name=None):
        """Returns either the number of charac variables within a source,
        or a dict {source_name: number of charac variables}, before splitting the sources.
        """
        if source_name is not None:
            return self._get_n_charac_variables()[source_name]
        return {
            # Each data variable within a source has its own charac variables
            source.name: source.n_charac_variables()
            for source in self.sources
        }

    def get_source_names(self):
        """Returns a list of the source names, including the split sources if needed."""
        return self.source_names

    def get_n_sources(self):
        """Returns the number of sources."""
        return len(self.get_source_names())

    def get_source_types_charac_vars(self):
        """Returns a dict {source_type: charac variables}."""
        # Browse all sources and collect their types and charac variables
        source_types_charac_vars = {}
        for source in self.sources:
            # If the source type has been seen before, make sure the charac
            # vars were the same. All sources from the same type should have
            # the same charac variables.
            if source.type in source_types_charac_vars:
                if source.charac_vars != source_types_charac_vars[source.type]:
                    raise ValueError(
                        f"Sources of type {source.type} have different charac variables."
                    )
            else:
                source_types_charac_vars[source.type] = source.charac_vars
        return source_types_charac_vars

    def get_n_charac_variables(self):
        """Returns a dict {source_name: number of charac variables}."""
        return self._get_n_charac_variables()
