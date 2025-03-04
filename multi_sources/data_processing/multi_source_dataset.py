"""Implements the MultiSourceDataset class."""

import torch
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from netCDF4 import Dataset
from pathlib import Path
from multi_sources.data_processing.grid_functions import crop_nan_border
from multi_sources.data_processing.source import Source
from multi_sources.data_processing.utils import compute_sources_availability, load_nc_with_nan
from multi_sources.data_processing.data_augmentation import MultisourceDataAugmentation


class MultiSourceDataset(torch.utils.data.Dataset):
    """A dataset that yields elements from multiple sources.
    A sample from the dataset is a dict {source_name: map}, where each map contains the
    following key-value pairs:
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

    For a given storm S and reference time t0,
    the dataset returns an element from each source that is between t0 and t0 - dt_max.
    When multiple elements are available for a source, a random one is selected.
    If a source is not available for the storm S at time t0, it is not included in the sample.
    """

    def __init__(
        self,
        dataset_dir,
        split,
        included_variables_dict,
        dt_max=24,
        min_available_sources=2,
        num_workers=0,
        select_most_recent=False,
        forecasting_lead_time=None,
        forecasting_sources=None,
        forecasting_source=None,
        include_seasons=None,
        exclude_seasons=None,
        randomly_drop_sources={},
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
            split (str): The split of the dataset to use. Must be one of "train", "val", or "test".
            included_variables_dict (dict): A dictionary
                {source_name: (list of variables, list of input-only variables)}
                containing the variables (i.e. channels) to include for each source.
                Variables also included in the second list will be included in the yielded
                data but will be flagged as input-only in the Source object.
            dt_max (int): The maximum time delta between the elements returned for each source,
                in hours.
            min_available_sources (int): The minimum number of sources that must be available
                for a sample to be included in the dataset. Does NOT include the forecasting source
                if it is enabled.
            num_workers (int): If > 1, number of workers to use for parallel loading of the data.
            select_most_recent (bool): If True, selects the most recent element for each source
                when multiple elements are available. If False, selects a random element.
            forecasting_lead_time (int): If not None, the lead time to use for forecasting.
                Must be used in conjunction with forecasting_sources. If enabled, the reference
                source will always be one of the forecast source.
                The time window for the other sources
                will be (t0 - lead_time - dt_max, t0 - lead_time), i.e. the other sources will be
                in a window of duration dt_max, ending at t0 - lead_time.
            forecasting_sources (list of str): If not None, the name of the sources to use
                for forecasting.
                Must be used in conjunction with forecasting_lead_time.
            include_seasons (list of int): The years to include in the dataset.
                If None, all years are included.
            exclude_seasons (list of int): The years to exclude from the dataset.
                If None, no years are excluded.
            randomly_drop_sources (dict of str to float): A dictionary {source_name: p}
                where p is the probability of dropping the source from the sample when
                it is available. A source cannot be dropped if it would result in a sample
                with less than min_available_sources sources.
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
        self.select_most_recent = select_most_recent
        self.min_available_sources = min_available_sources
        self.randomly_drop_sources = randomly_drop_sources
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
        for source_name, (all_vars, input_only_vars) in self.variables_dict.items():
            # Update the source metadata with the included variables
            sources_metadata[source_name]["data_vars"] = all_vars
            sources_metadata[source_name]["input_only_vars"] = input_only_vars
            # Create the source object
            self.sources.append(Source(**sources_metadata[source_name]))
            self.sources_dict[source_name] = self.sources[-1]
        # Load the samples metadata based on the split
        self.df = pd.read_csv(self.dataset_dir / f"{split}.csv", parse_dates=["time"])

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

        if forecasting_lead_time is not None:
            if forecasting_source is not None:
                forecasting_sources = [forecasting_source]
            if forecasting_sources is None:
                raise ValueError("forecasting_lead_time must be used with forecasting_sources.")
            self.forecasting_lead_time = pd.Timedelta(forecasting_lead_time, unit="h")
            self.forecasting_sources = [s for s in forecasting_sources if s in source_names]
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
        # If forecasting, don't include the forecast in the available sources, since they
        # won't be included in the samples.
        if self.forecasting_sources is not None:
            available_sources = available_sources.drop(columns=self.forecasting_sources)
        # - From this, we can filter the samples to only keep the ones where at least
        # min_available_sources sources are available.
        available_sources_count = available_sources.sum(axis=1)
        # We can now build the reference dataframe:
        # - self.df contains the metadata for all elements from all sources,
        # - self.reference_df is a subset of self.df containing every (sid, time) pair
        #   that defines a sample for which at least min_available_sources sources are available.
        mask = available_sources_count >= min_available_sources
        self.reference_df = self.df[mask][["sid", "time", "source_name"]]
        self.reference_df["n_available_sources"] = available_sources_count[mask]

        # If forecasting, only use the forecasting sources as references
        if self.forecasting_sources is not None:
            self.reference_df = self.reference_df[
                self.reference_df["source_name"].isin(self.forecasting_sources)
            ]

        # Finally, avoid duplicated samples if two observations are at the exact same time
        self.reference_df = self.reference_df.drop_duplicates(["sid", "time"])
        self.reference_df = self.reference_df.reset_index(drop=True)

        # ========================================================================================
        # Pre-computation to speed up the data loading
        # Map {sid: subset of self.df}
        print(f"{split}: Pre-computing the sid to dataframe mapping...")
        self.sid_to_df = {}
        for sid in self.reference_df["sid"].unique():
            self.sid_to_df[sid] = self.df[self.df["sid"] == sid]

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
            sample (multi_sources.data_processing.multi_source_batch.MultiSourceBatch): The element
                at the given index.
        """
        sample = self.reference_df.iloc[idx]
        sid, t0 = sample["sid"], sample["time"]
        # Isolate the rows of self.df corresponding to the sample sid
        sample_df = self.sid_to_df[sid]
        # For each source, try to load the element at the given time
        output = {}
        for source in self.sources:
            source_name = source.name
            # Random dropping: check if the source should be dropped
            if source_name in self.randomly_drop_sources:
                p = self.randomly_drop_sources[source_name]
                if np.random.rand() < p:
                    # Skip the source
                    continue

            # If forecast source: if it is the reference source, include it;
            # otherwise, skip it.
            if self.forecasting_sources is not None and source_name in self.forecasting_sources:
                if source_name == sample["source_name"]:
                    # Isolate the rows of sample_df corresponding to the right source
                    # and where the time is within the time window
                    min_t = t0 - self.dt_max
                    max_t = t0
                    df = sample_df[
                        (sample_df["source_name"] == source_name) & (sample_df["time"] == t0)
                    ]
                else:
                    continue
            else:
                # Isolate the rows of sample_df corresponding to the right source
                # and where the time is within the time window
                min_t = t0 - self.forecasting_lead_time - self.dt_max
                max_t = t0 - self.forecasting_lead_time
                df = sample_df[
                    (sample_df["source_name"] == source_name)
                    & (sample_df["time"] <= max_t)
                    & (sample_df["time"] > min_t)
                ]

            if self.select_most_recent:
                # Sort by descending time so that the first row is the closest to t0
                df = df.sort_values("time", ascending=False)
            else:
                # Shuffle the rows so that the first row is random
                df = df.sample(frac=1)
            # Check if there is at least one element available for this source
            if len(df) > 0:
                time = df["time"].iloc[0]
                dt = t0 - self.forecasting_lead_time - time
                DT = torch.tensor(
                    dt.total_seconds() / self.dt_max.total_seconds(),
                    dtype=torch.float32,
                )

                # Load the npy file containing the data for the given storm, time, and source
                filepath = Path(df["data_path"].iloc[0])
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

                    # The values can contain borders that are fully NaN (due to the sources
                    # originally having multiple channels that are not aligned geographically).
                    # Compute how much we can crop them, and apply that cropping to all spatial
                    # tensors to keep the spatial alignment.
                    V, C, LM, D = crop_nan_border(V, [V, C, LM, D])

                    # Normalize the characs and values tensors
                    CA, V = self.normalize(V, source, CA)
                    output_entry = {
                        "dt": DT,
                        "coords": C,
                        "landmask": LM,
                        "dist_to_center": D,
                        "values": V,
                    }
                    if CA is not None:
                        output_entry["characs"] = CA
                    output[source_name] = output_entry
        # (Optional) Data augmentation
        if isinstance(self.data_augmentation, MultisourceDataAugmentation):
            output = self.data_augmentation(output)

        if len(output) <= 1:
            raise ValueError(
                f"Sample {sid} at time {t0} has only {len(output)} sources available."
                " Check the parameters of the experiment; random sources dropping might"
                " make leave only 1 or 0 source available. "
            )
        return output

    def normalize(
        self,
        values,
        source,
        characs=None,
        denormalize=False,
        batched=False,
        dist_to_center=False,
        device=None,
    ):
        """Normalizes the characs and values tensors associated with a given
        source, and optionally a specific data variable.
        Args:
            values (torch.Tensor): tensor of shape (C, ...) if dvar is None,
                or (1, ...) if dvar is specified, where ... are the spatial dimensions.
            source (Source or str): Source object representing the source, or name
                of the source.
            characs (torch.Tensor, optiona): tensor of shape (n_charac_vars,)
                containing the characteristic variables.
            denormalize (bool, optional): If True, denormalize the characs and values tensors
                instead of normalizing them.
            batched (bool, optional): If True, the values tensor is expected to have
                shape (B, C, ...), where B is the batch size.
            dist_to_center (bool, optional): expects the values tensor to have C + 1 channels,
                where C is the number of data variables in the source, and the additional
                channel is the distance to the center of the storm.
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

        data_vars = source.data_vars
        if dist_to_center:
            data_vars = data_vars + ["dist_to_center"]

        data_means = np.array([self.data_means[source_name][var] for var in data_vars]).reshape(
            -1, *(1 for _ in (values.shape[1:] if not batched else values.shape[2:]))
        )  # (C, 1, ...) like values
        data_stds = np.array([self.data_stds[source_name][var] for var in data_vars]).reshape(
            -1, *(1 for _ in (values.shape[1:] if not batched else values.shape[2:]))
        )
        data_means = torch.tensor(data_means, dtype=values.dtype)
        data_stds = torch.tensor(data_stds, dtype=values.dtype)

        if device is not None:
            data_means = data_means.to(device)
            data_stds = data_stds.to(device)
        if denormalize:
            normalized_values = values * data_stds + data_means
        else:
            normalized_values = (values - data_means) / data_stds

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
