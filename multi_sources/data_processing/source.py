"""
Implements the Source class.
"""
import yaml
import os


class Source:
    """Represents a source of data, which could be 2D, 1D or 0D (scalar), not accounting for the
    channel dimension.
    
    Attributes:
        name (str): The name of the source, under the format
            "source.subsource1.subsource2. ... .subsourceN".
        dims (int): The number of dimensions of the source.
        variables (list of str): The names of the variables in the source, i.e.
            variables that can be found in a netCDF file of the source.
        env_vars (list of str): Same as variables, but for environment variables (e.g.
            latitude, longitude, month).
    """
    def __init__(self, name, dims, variables, env_vars):
        self.name = name
        self.dims = dims
        self.variables = variables
        self.env_vars = env_vars

    def get_path(self):
        """Returns the path to the root directory of the source."""
        # Load the paths config file to get the sources directory
        with open("paths.yml", "r") as file:
            paths = yaml.safe_load(file)
        # The path within the root dir is just the name of the source, where dots are replaced by slashes.
        return os.path.join(paths['sources'], self.name.replace(".", "/"))

    def __repr__(self):
        return f"Source(name={self.name}, \n\
                 dims={self.dims}, \n\
                 variables={self.variables}, \n\
                 env_vars={self.env_vars})"
