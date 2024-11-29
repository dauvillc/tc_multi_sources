"""Implements small functions for preprocessing."""

import netCDF4 as nc


def list_tc_primed_sources(tc_primed_path, exclude_years=None, source_type="all"):
    """Recursively find all source files from TC-PRIMED.

    Args:
        tc_primed_path (Path): Path to the root directory of TC-PRIMED.
        exclude_years (list of int, optional): List of years to exclude from the search.
        source_type (str, optional): Type of sources to return. One of:
            - "all": Return all sources (default)
            - "satellite": Return only satellite sources (radar and pmw)
            - "environmental": Return only environmental sources (era5)

    Returns:
        sources (list of str): List of source names ({sensor}_{satellite} or "era5").
        source_files (dict of str: list of Path): Dictionary mapping each source
            to the list of corresponding source files.
        source_groups (dict of str: list of str): Dictionary mapping each source
            to the list of corresponding groups. For example, ['passive_microwave', 'S4']
            means that the data for the source can be found in ds['passive_microwave']['S4'].
    """
    storm_files = list_tc_primed_storm_files(tc_primed_path, exclude_years)
    overpass_files, environment_files = storm_files

    # Get all files
    all_overpass_files = []
    for files in overpass_files.values():
        all_overpass_files.extend(files)
    all_env_files = []
    for files in environment_files.values():
        all_env_files.extend(files)

    # Get sensor_satellite pairs from overpass files
    sensat_pairs = set()
    for file in all_overpass_files:
        sensat_pairs.add("_".join(file.stem.split("_")[3:5]))
    sensat_pairs = sorted(list(sensat_pairs))

    # Create source_files and source_groups
    source_files, source_groups = {}, {}
    all_sources = []

    # Handle ERA5 files
    if source_type in ["all", "environmental"]:
        all_sources.append("era5")
        source_files["era5"] = all_env_files
        source_groups["era5"] = ["rectilinear"]

    if source_type in ["all", "satellite"]:
        # Handle passive microwave and radar sources, whose data is either in the
        # "passive_microwave" or "radar_radiometer" group.
        # Within that group, the data is further divided into swaths (e.g., 'S4', 'S5').
        # Each swath is considered a separate source.
        for sensat_pair in sensat_pairs:
            # - Open the first file of the pair to get the swaths
            sensat_files = [file for file in all_overpass_files if sensat_pair in file.stem]
            with nc.Dataset(sensat_files[0], "r") as ds:
                # PMW swaths
                pmw_swaths = [gp for gp in ds["passive_microwave"].groups]
                # Radar swaths
                radar_swaths = []
                if "radar_radiometer" in ds.groups:
                    radar_swaths = [gp for gp in ds["radar_radiometer"].groups]
            # - Create a source for each swath
            for swath in pmw_swaths:
                source = f"pmw_{sensat_pair}_{swath}"
                source_files[source] = [
                    file for file in all_overpass_files if sensat_pair in file.stem
                ]
                source_groups[source] = ["passive_microwave", swath]
                all_sources.append(source)
            for swath in radar_swaths:
                source = f"radar_{sensat_pair}_{swath}"
                source_files[source] = [
                    file for file in all_overpass_files if sensat_pair in file.stem
                ]
                source_groups[source] = ["radar_radiometer", swath]
                all_sources.append(source)

    return all_sources, source_files, source_groups


def list_tc_primed_storm_files(tc_primed_path, exclude_years=None):
    """Lists all source files for all storms in TC-PRIMED.

    Args:
        tc_primed_path (Path): Path to the root directory of TC-PRIMED.
        exclude_years (list of int, optional): List of years to exclude from the search.

    Returns:
        tuple: A pair of dictionaries (overpass_files, environment_files), where each dictionary maps
            (year, basin, number) to the list of corresponding source files.
            - overpass_files: Contains satellite overpass observation files
            - environment_files: Contains environmental condition files
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

    # Separate overpass and environment files
    overpass_files = {}
    environment_files = {}
    for key, files in storm_files.items():
        overpass_files[key] = [
            file
            for file in files
            if "era5" not in file.stem and "env" not in file.stem and file.suffix == ".nc"
        ]
        environment_files[key] = [
            file
            for file in files
            if ("era5" in file.stem or "env" in file.stem) and file.suffix == ".nc"
        ]

    return overpass_files, environment_files
