"""Implements small utility functions for data processing."""

import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def _get_leaf_subsources(source_dict, path="", previous_vars=[]):
    """Returns the leaf subsources of a source dictionary."""
    # Recursivity stop condition: if no subsource key is found, return the source
    # with its varables as well as the previous ones. Replace
    # the dim key if it is already present.
    subsource_keys = [key for key in source_dict.keys() if key != "variables"]
    if not subsource_keys:
        return {path: (previous_vars + source_dict.get("variables", []))}
    # If there are subsource keys, call the function recursively on each subsource.
    returned_dict = {}
    for subsource_key in subsource_keys:
        returned_dict.update(
            _get_leaf_subsources(
                source_dict[subsource_key],
                path + "_" + subsource_key,  # source_subsource_ ... _lastsubsource
                previous_vars + source_dict.get("variables", []),
            )
        )
    return returned_dict


def read_variables_dict(variables_dict):
    """Reads the variables dictionary that specifies which variables should be
    included from which source.
    Args:
        variables_dict: dictionary with the following structure:
            {
                "source1": {
                    "subsource1": {
                        "variables": ["var1", "var2", ...],
                        "subsource2": {
                            "variables": ["var3", "var4", ...],
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
            {
                "source1_subsource1_subsource2_..._lastsubsource": ["var1", "var2", ...],
                ...
            }
    """
    result = _get_leaf_subsources(variables_dict)
    # Remove the initial '_' at the beginning of each source key.
    return {key[1:]: value for key, value in result.items()}


def _is_source_available(ref_obs, sid_mask, source_mask, time_arr, source_name, dt_max):
    """Given a row in the metadata df which defines a sample (reference
    observation) and a source, returns 1 if the source is present in the
    sample and 0 otherwise."""
    t0 = ref_obs["time"]
    sid = ref_obs["sid"]
    # Compute the oldest time an observation can have to respect the time delta constraint.
    min_t = t0 - dt_max
    # Isolate the times of observations corresponding to the correct sid and source
    times = time_arr[sid_mask[sid] & source_mask]
    # Check for times that respect the time delta constraint
    return int(((times <= t0) & (times > min_t)).any())


def _source_availability(df, sid_mask, source_mask, time_arr, source_name, dt_max):
    """Returns a series S with one row per sample,
    such that S[i] is 1 if source_name is available at sample i and 0 otherwise."""
    return df.apply(
        _is_source_available,
        args=(sid_mask, source_mask, time_arr, source_name, dt_max),
        axis="columns",
    ).to_numpy()


def compute_sources_availability(df, dt_max, num_workers=0):
    """Given a DataFrame whose rows correspond to samples from varying sources,
    computes which sources are available for each sample.
    For a sample (s0, sid0, t0), a source s1 is considered available if there is
    at least one sample (s1, sid0, t1) such that t0 - dt_max <= t1 <= t0.
    Args:
        df (pd.DataFrame): DataFrame including at least the columns 'source_name',
            'sid' and 'time'.
        dt_max (pd.Timedelta): Time window to consider sources as available.
        num_workers (int, optional): Number of workers to use for parallelization.

    Returns:
        pd.DataFrame: DataFrame D with one column per source, where D.loc[i, s] = 1
            if source s is available for the sample i and 0 otherwise.
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
                )
            for source, future in futures.items():
                result[source] = future.result()
    else:
        for source in df["source_name"].unique():
            result[source] = _source_availability(
                df, sid_mask, source_mask[source], time_arr, source, dt_max
            )
    return pd.DataFrame(result)
