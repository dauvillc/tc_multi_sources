"""
Implements the Source class.
"""


class Source:
    """Represents a source of data, which could be 2D, 1D or 0D (scalar), not accounting for the
    channel dimension.

    Attributes:
        name (str): The name of the source, under the format
            "source.subsource1.subsource2. ... .subsourceN".
        source_type (str): The type of the source, e.g. "passive_microwave".
        dim (int): The number of dimensions of the source.
        shape (tuple of int): The shape of the source, i.e. the number
            of elements in each dimension.
        data_vars (list of str): The names of the variables in the source, i.e.
            variables that can be found in a netCDF file of the source.
        context_vars (list of str): The names of the context variables in the source, e.g.
            frequency, IFOV, etc.
    """

    def __init__(self, source_name, source_type, dim, shape, data_vars, context_vars, **kwargs):
        if len(shape) != dim:
            raise ValueError("The number of dimensions must match the length of shape.")
        self.name = source_name
        self.dim = dim
        self.shape = shape
        self.source_type = source_type
        self.data_vars = data_vars
        self.context_vars = context_vars

    def n_data_variables(self):
        """Returns the number of data variables in the source."""
        return len(self.data_vars)

    def n_context_variables(self):
        """Returns the number of context variables in the source."""
        return len(self.context_vars)

    def __repr__(self):
        return f"Source(name={self.name}, \n\
                 dim={self.dim}, \n\
                 data vars={self.data_vars}, \n\
                 context vars={self.context_vars})"
