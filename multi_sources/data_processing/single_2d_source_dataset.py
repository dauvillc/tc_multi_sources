"""Implements the SingleSourceDataset class."""
import xarray as xr
import torch
from pathlib import Path
from torch.utils.data import Dataset


class Single2DSourceDataset(Dataset):
    """A dataset that contains samples from a single source of 2D data (e.g. microwave
    observations).
    """
    def __init__(self, source):
        """
        Args:
            source: multi_sources.data_processing.source.Source object
        """
        self.source = source
        self.data_dir = Path(self.source.get_path())
        # Load in memory the normalization statistics
        self.normalization_mean = xr.open_dataset(self.data_dir / 'normalization_mean.nc')
        self.normalization_std = xr.open_dataset(self.data_dir / 'normalization_std.nc')
        # Recursively find all paths to the data files
        self.data_files = []
        for season in self.data_dir.iterdir():
            if not season.is_dir():
                continue
            for basin in season.iterdir():
                for data_file in basin.glob('*.nc'):
                    self.data_files.append(data_file)
        print(f'Found {len(self.data_files)} storm data files.')
        # Lazy-load the files in parallel. All timesteps from all storms are concatenated along the
        # 'sample' dimension.
        self.data = xr.open_mfdataset(self.data_files, combine='nested', concat_dim="sample", parallel=False)

    def __getitem__(self, idx):
        # Isolate the data for the given sample
        data = self.data.isel(sample=idx)
        # Retrieve the list of variables from the source
        variables = self.source.variables
        # Select the variables from the data and convert them to a tensor
        data = data[variables].to_array().load().data
        return torch.tensor(data)

    def __len__(self):
        return len(self.data.time)

