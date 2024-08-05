"""Implements small functions for preprocessing."""

import netCDF4 as nc


def list_tc_primed_sources(tc_primed_path, exclude_years=None):
    """Recursively find all source files from TC-PRIMED.

    Args:
        tc_primed_path (Path): Path to the root directory of TC-PRIMED.
        exclude_years (list of int, optional): List of years to exclude from the search.

    Returns:
        sen_sat_pairs (list of str): List of strings {sensor}.{satellite}.
        sen_sat_files (dict of str: list of Path): Dictionary mapping each {sensor}.{satellite}
            to the list of corresponding source files.
        sen_sat_swaths (dict of str: list of str): Dictionary mapping each {sensor}.{satellite}
            to the list of corresponding swaths.
    """
    storm_files = list_tc_primed_storm_files(tc_primed_path, exclude_years)
    # Concatenate all the files
    overpass_files = []
    for files in storm_files.values():
        overpass_files.extend(files)
    # Deduce the list of strings {sensor}_{satellite} from the filenames
    sen_sat_pairs = set()
    for file in overpass_files:
        sen_sat_pairs.add("_".join(file.stem.split("_")[3:5]))
    sen_sat_pairs = sorted(list(sen_sat_pairs))
    sen_sat_files, sen_sat_swaths = {}, {}
    for sensat in sen_sat_pairs:
        # Retrieve the list of files whose stem contains the sensor/satellite pair
        sen_sat_files[sensat] = [file for file in overpass_files if sensat in file.stem]
        # We need to open a file with netCDF4 to retrieve the list of swaths
        with nc.Dataset(sen_sat_files[sensat][0], "r") as ds:
            swaths = [swath for swath in ds["passive_microwave"].groups.keys()]
        # If the satellite is GPM or TRMM, we'll also retrieve the radar-radiometer
        # data and consider them as swaths
        if "GPM" in sensat:
            swaths.extend(['KuGMI'])
        elif "TRMM" in sensat:
            swaths.extend(['KuTMI'])
        sen_sat_swaths[sensat] = swaths
    return sen_sat_pairs, sen_sat_files, sen_sat_swaths


def list_tc_primed_storm_files(tc_primed_path, exclude_years=None):
    """Lists all source files for all storms in TC-PRIMED.

    Args:
        tc_primed_path (Path): Path to the root directory of TC-PRIMED.
        exclude_years (list of int, optional): List of years to exclude from the search.

    Returns:
        storm_files (dict of (str, str, str): list of Path): Dictionary mapping each
            (year, basin, number) to the list of corresponding source files.
    """
    if exclude_years is None:
        exclude_years = []
    # The raw dataset has the structure tc_primed/{year}/{basin}/{number}/{filename}.nc
    storm_files = {}
    for year in tc_primed_path.iterdir():
        if int(year.stem) in exclude_years:
            continue
        for basin in year.iterdir():
            for number in basin.iterdir():
                storm_files[(year.stem, basin.stem, number.stem)] = list(number.iterdir())
    # The filenames are formatted as:
    # - TCPRIMED_VERSION_BASINNUMBERYEAR_SENSOR_SATELLITE_IMGNUMBER_YYYYMMDDHHmmSS.nc
    #   for the overpass files;
    # - TCPRIMED_VERSION_BASINNUMBERYEAR_era5_START_END.nc for the environmental files.
    # Isolate the overpass files:
    overpass_files = {}
    for key, files in storm_files.items():
        overpass_files[key] = [file for file in files if "era5" not in file.stem]
    return overpass_files
