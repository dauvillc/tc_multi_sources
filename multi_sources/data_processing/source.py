"""
Implements the Source class.
"""


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

    def n_variables(self):
        """Returns the number of variables in the source."""
        return len(self.variables)

    def __repr__(self):
        return f"Source(name={self.name}, \n\
                 dims={self.dims}, \n\
                 variables={self.variables}, \n\
                 env_vars={self.env_vars})"
