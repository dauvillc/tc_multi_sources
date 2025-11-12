"""Implements small utility functions for data processing."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


def _get_leaf_subsources(
    source_dict, path="", previous_vars=[], previous_input_only=[], previous_output_only=[]
):
    """Returns the leaf subsources of a source dictionary."""
    # Recursivity stop condition: if no subsource key is found, return the source
    # with its varables as well as the previous ones. Replace
    # the dim key if it is already present.
    subsource_keys = [
        key for key in source_dict.keys() if key not in ["variables", "input_only", "output_only"]
    ]
    if not subsource_keys:
        return {
            path: (
                previous_vars + source_dict.get("variables", []),
                previous_input_only + source_dict.get("input_only", []),
                previous_output_only + source_dict.get("output_only", []),
            )
        }
    # If there are subsource keys, call the function recursively on each subsource.
    returned_dict = {}
    for subsource_key in subsource_keys:
        returned_dict.update(
            _get_leaf_subsources(
                source_dict[subsource_key],
                path + "_" + subsource_key,  # source_subsource_ ... _lastsubsource
                previous_vars + source_dict.get("variables", []),
                previous_input_only + source_dict.get("input_only", []),
                previous_output_only + source_dict.get("output_only", []),
            )
        )
    return returned_dict


def read_variables_dict(variables_dict):
    """Reads the variables dictionary that specifies which variables should be
    included from which source, as well as which variables are input-only.
    Args:
        variables_dict: dictionary with the following structure:
            {
                "source1": {
                    "subsource1": {
                        "variables": ["var1", "var2", ...],
                        "input_only": ["var1", "var2", ...],
                        "output_only": ["var1", "var2", ...],
                        "subsource2": {
                            "variables": ["var3", "var4", ...],
                            "input_only": ["var3", "var4", ...],
                            ...
                        },
                        ...
                    },
                    ...
                },
                "source2": {
                    ...
                },
                ...
            }
    Returns:
        A dictionary with the following structure:
        variables_dict:
            {
                "source1_subsource1_subsource2_..._lastsubsource":
                    (["var1", "var2", ...],
                    ['input_only_var_1', 'input_only_var_2', ...],
                    ['output_only_var_1', 'output_only_var_2', ...],
                    ),
                ...
            }
    """
    result = _get_leaf_subsources(variables_dict)
    # Remove the initial '_' at the beginning of each source key.
    return {key[1:]: value for key, value in result.items()}


def _is_source_available(ref_obs, sid_mask, source_mask, time_arr, source_name, dt_max, lead_time):
    """Given a row in the metadata df which defines a sample (reference
    observation) and a source, returns the number of observations from that source
    that are available for that sample."""
    t0 = ref_obs["time"]
    sid = ref_obs["sid"]
    # Compute the allowed time window
    min_t = t0 - lead_time - dt_max
    max_t = t0 - lead_time
    # Isolate the times of observations corresponding to the correct sid and source
    times = time_arr[sid_mask[sid] & source_mask]
    # Check for times that respect the time delta constraint
    return ((times > min_t) & (times <= max_t)).sum().item()


def _source_availability(df, sid_mask, source_mask, time_arr, source_name, dt_max, lead_time):
    """Returns a series S with one row per sample,
    such that S[i] is 1 if source_name is available at sample i and 0 otherwise."""
    return df.apply(
        _is_source_available,
        args=(sid_mask, source_mask, time_arr, source_name, dt_max, lead_time),
        axis="columns",
    ).to_numpy()


def compute_sources_availability(df, dt_max, lead_time, num_workers=0):
    """Given a DataFrame whose rows correspond to samples from varying sources,
    computes which sources are available for each sample.
    For a sample (s0, sid0, t0), a source s1 is considered available if there is
    at least one sample (s1, sid0, t1) such that t0 - lt - dt_max < t1 <= t0 - lt.
    Args:
        df (pd.DataFrame): DataFrame including at least the columns 'source_name',
            'sid' and 'time'.
        dt_max (pd.Timedelta): Time window to consider sources as available.
        lead_time (pd.Timedelta): Lead time to consider when computing the
            availability of sources.
        num_workers (int, optional): Number of workers to use for parallelization.

    Returns:
        pd.DataFrame: DataFrame D with one column per source, where D.loc[i, s] is
            the number of available observations from source s for sample i.
    """
    # Isolate the time, sid and source columns
    df = df[["time", "sid", "source_name"]]
    # We'll work with numpy arrays for speed
    time_arr = df["time"].to_numpy()
    sid_col = df["sid"].to_numpy()
    source_col = df["source_name"].to_numpy()
    # Create a mask for each sid and source
    sid_mask = {sid: sid_col == sid for sid in df["sid"].unique()}
    source_mask = {source: source_col == source for source in df["source_name"].unique()}
    # Initialize the result DataFrame
    result = {}
    # Process the sources either in parallel or sequentially
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for source in df["source_name"].unique():
                futures[source] = executor.submit(
                    _source_availability,
                    df,
                    sid_mask,
                    source_mask[source],
                    time_arr,
                    source,
                    dt_max,
                    lead_time=lead_time,
                )
            for source, future in futures.items():
                result[source] = future.result()
    else:
        for source in df["source_name"].unique():
            result[source] = _source_availability(
                df, sid_mask, source_mask[source], time_arr, source, dt_max, lead_time=lead_time
            )
    return pd.DataFrame(result)


def load_nc_with_nan(netcdf_var):
    """Loads a netCDF variable into a numpy array, filling the masked values with NaN.
    Args:
        netcdf_var (netCDF4.Variable): Variable to load.
    Returns:
        np.ndarray: Numpy array with the variable values.
    """
    return netcdf_var[:].filled(np.nan)
