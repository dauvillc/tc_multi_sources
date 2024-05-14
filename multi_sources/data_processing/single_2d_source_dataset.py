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
        self.normalization_mean = xr.open_dataset(self.data_dir / "normalization_mean.nc")
        self.normalization_std = xr.open_dataset(self.data_dir / "normalization_std.nc")
        # The data files are stored as season.nc (e.g. 2018.nc)
        self.data_files = [
            file
            for file in self.data_dir.glob("*.nc")
            if file.name != "normalization_mean.nc" and file.name != "normalization_std.nc"
        ]
        print(f"Found {len(self.data_files)} storm data files.")
        # Lazy-load the files in parallel. All timesteps from all storms are concatenated along the
        # 'sample' dimension.
        self.data = xr.open_mfdataset(
            self.data_files, combine="nested", concat_dim="sample", parallel=False,
            chunks={'sample': 1}
        )
        # Load the coordinates in memory for indexing
        self.data = self.data.assign_coords(
            {coord: self.data[coord].load() for coord in self.data.coords}
        )
        # Create a coordinate "SID" that uniquely identifies each storm, as a string
        # SEASONBASINCYCLONE_NUMBER
        self.data["SID"] = (
            self.data["season"]
            .astype(str)
            .str.cat(self.data["basin"], self.data["cyclone_number"].astype(str), sep="")
        )
        # Set the "SID" and time as indexes
        self.data = self.data.set_index(sample=["SID", "time"])

    def timesteps_dataframe(self):
        """Returns a pandas DataFrame giving the times at which elements are available
        for each storm. Its columns are 'SID', 'season', 'basin', 'cyclone_number', and 'time'.
        """
        return (
            self.data[["SID", "season", "basin", "cyclone_number", "time"]]
            .reset_index("sample")
            .to_dataframe()
        )

    def get_sample(self, sid, t):
        """Returns the data for the given storm at the given time.

        Args:
            sid (str): The storm identifier (STORMBASINCYCLONE_NUMBER).
            t (np.datetime64): The time of the desired data.
        Returns:
            C (np.ndarray): array of shape (2, H, W) containing the latitude and longitude
                coordinates at each pixel.
            D (np.ndarray): array of shape (H, W) containing the distance to the storm center
                in km at each pixel.
            V (np.ndarray): array of shape (n_vars, H, W) containing the variables at each pixel.
        Raises:
            KeyError: if the entry does not exist at the given coordinates.
        """
        # Isolate the data in self.data for the given storm and time
        data = self.data.sel(sample=sid).sel(time=t)
        # Retrieve the list of variables from the source
        variables = self.source.variables
        # Select the variables from the data
        V = data[variables].to_array().load().values
        # Select the coordinates
        C = data[["latitude", "longitude"]].to_array().values
        # Compute the distance to the storm center from the x and y coordinates
        D = data["x"].values ** 2 + data["y"].values ** 2
        D = D**0.5
        return C, D, V

    def __getitem__(self, idx):
        # Isolate the data for the given sample
        data = self.data.isel(sample=idx)
        # Retrieve the list of variables from the source
        variables = self.source.variables
        # Select the variables from the data and convert them to a tensor
        data = data[variables].to_array().load().data
        return torch.tensor(data)

    def get_spatial_size(self):
        """Returns the spatial size of the data (H, W)."""
        return (size for dim, size in self.data.sizes.items() if dim != "sample")

    def get_n_variables(self):
        """Returns the number of variables in the data."""
        return self.source.n_variables()

    def __len__(self):
        return len(self.data.time)
