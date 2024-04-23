"""
Clément Dauvilliers - 23/04/2024.
Preprocesses the TC-Primed dataset by normalizing the data, treating missing values,
and formatting the result into the sources tree.
"""

import yaml
import xarray as xr
import argparse
import netCDF4 as nc
from pathlib import Path
from tqdm import tqdm


def pad_dataset(ds, max_size):
    """
    Pads the dataset ds to the size max_size, by adding missing values
    symmetrically on the sides of the image.
    """
    # Retrieve the size of the dataset
    sizes = ds.sizes
    size = (sizes['scan'], sizes['pixel'])
    # Compute the padding to add on each side
    pad_top = (max_size[0] - size[0]) // 2
    pad_bottom = max_size[0] - size[0] - pad_top
    pad_left = (max_size[1] - size[1]) // 2
    pad_right = max_size[1] - size[1] - pad_left
    # Pad the dataset with the fill value -9999.9 (from the tc_primed documentation)
    # Coordinates are padded with NaN
    ds = ds.pad(scan=(pad_top, pad_bottom), pixel=(pad_left, pad_right),
                mode='constant', constant_values=-9999.9)
    return ds


def main():
    # Load the paths configuration file
    with open("paths.yml", "r") as file:
        paths_cfg = yaml.safe_load(file)
    tc_primed_path = Path(paths_cfg["raw_datasets"]) / "tc_primed"
    dest_path = Path(paths_cfg["sources"]) / "tc_primed"
    # Load the sources configuration file
    with open("sources.yml", "r") as file:
        sources_cfg = yaml.safe_load(file)["tc_primed"]
        sen_sat_pairs = sources_cfg["sensor_satellite_pairs"]
        # The sen/sat pairs are dicts of the form
        # SENSOR_SATELLITE: {swath1: [bands], swath2: [bands], ...}
        # If sen_sat_pairs is None, process all sensor/satellite pairs
    # The sources configuration contains the list of pairs (sensor, satellite) to include.
    # The raw dataset has the structure tc_primed/{year}/{basin}/{number}/{filename}.nc
    # Retrieve all filenames for all years
    all_files = []
    for year in tc_primed_path.iterdir():
        for basin in year.iterdir():
            for number in basin.iterdir():
                all_files.extend(number.glob("*.nc"))
    # The filenames are formatted as
    # TCPRIMED_VERSION_BASINNUMBERYEAR_SENSOR_SATELLITE_IMGNUMBER_YYYYMMDDHHmmSS.nc
    # Deduce the list of strings {sensor}_{satellite} from the filenames
    available_sen_sat = set()
    for file in all_files:
        available_sen_sat.add("_".join(file.stem.split("_")[3:5]))
    # If the list of pairs is not specified, process all available pairs
    # and for each pair, process all swaths and bands
    if sen_sat_pairs is None:
        sen_sat_pairs = {sensat: None for sensat in available_sen_sat}
    # Otherwise, check that the pairs in the configuration file are in the dataset
    else:
        for pair in sen_sat_pairs.keys():
            if pair not in available_sen_sat:
                raise ValueError(f"Sensor/Satellite pair {pair} not found in the dataset.")
    # For each sensor/satellite pair, preprocess the data
    for sensat in sen_sat_pairs.keys():
        print(f"Processing sensor/satellite pair {sensat}")
        # Retrieve the list of files whose stem contains the sensor/satellite pair
        files = [file for file in all_files if sensat in file.stem]
        # If the list of swaths is not specified, we need to open a file with netCDF4 to
        # retrieve the list of swaths
        if sen_sat_pairs[sensat] is None:
            with nc.Dataset(files[0], "r") as ds:
                swaths = [swath for swath in ds['passive_microwave'].groups.keys()]
        # Otherwise, use the list of swaths in the configuration file
        else:
            swaths = list(sen_sat_pairs[sensat].keys())
        # For each swath, preprocess the data
        for swath in swaths:
            print(f"Processing swath {swath}")
            # Although a swath always contains the same bands, the size of the images is not
            # identical across the samples, due to missing values that have been removed.
            # Therefore, we can't stack the images into a single xarray dataset just yet.
            # To do so, we'll pad all images from the same swath to the same size. That size
            # is the maximum size of all images from the swath.
            # - Retrieve the maximum size of the images from the swath
            max_size = [0, 0]
            print("Retrieving maximum size of images")
            for file in tqdm(files):
                with nc.Dataset(file, "r") as ds:
                    dims = ds['passive_microwave'][swath].dimensions
                    size = (dims['scan'].size, dims['pixel'].size)
                    max_size[0] = max(max_size[0], size[0])
                    max_size[1] = max(max_size[1], size[1])
            # - Load all images after padding them via the preprocessing function
            print("Loading images")
            def _preprocess(ds):
                return pad_dataset(ds, max_size)
            dataset = xr.open_mfdataset(files, preprocess=_preprocess,
                                        concat_dim='sample', combine='nested')
            return 0

if __name__ == "__main__":
    main()
