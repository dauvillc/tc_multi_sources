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
        dim (int): The number of dimensions of the source (excluding the batch dim)
        shape (tuple of int): The shape of the source, i.e. the number
            of elements in each dimension, *excluding* the batch and channel dimensions.
        data_vars (list of str): The names of the variables in the source, i.e.
            variables that can be found in a netCDF file of the source.
        context_vars (list of str): The names of the context variables in the source, e.g.
            frequency, IFOV, etc.
        input_only_vars (list of str, optional): The names of the variables that are input-only,
            which means they are not trained on. Defaults to [], i.e. all variables are both
            used as input and target.
    """

    def __init__(
        self,
        source_name,
        source_type,
        dim,
        shape,
        data_vars,
        context_vars,
        input_only_vars=[],
        **kwargs,
    ):
        if (dim == 0 and shape != [1]) or (dim > 0 and len(shape) != dim):
            raise ValueError("The number of dimensions must match the length of shape.")
        self.name = source_name
        self.dim = dim
        self.shape = shape
        self.type = source_type
        self.data_vars = data_vars
        self.context_vars = context_vars
        self.input_only_vars = input_only_vars
        self.output_vars = [var for var in data_vars if var not in input_only_vars]
        # Make sure the input-only variables are in the data variables
        for var in self.input_only_vars:
            if var not in self.data_vars:
                raise ValueError(f"Input-only variable {var} not found in data variables.")
        # Pre-compute the output variables mask
        self.output_vars_mask = [var not in self.input_only_vars for var in self.data_vars]

    def n_data_variables(self):
        """Returns the number of data variables in the source."""
        return len(self.data_vars)

    def n_input_variables(self):
        """All data variables are input variables."""
        return self.n_data_variables()

    def n_target_variables(self):
        """Returns the number of target variables in the source."""
        return self.n_data_variables() - len(self.input_only_vars)

    def n_context_variables(self):
        """Returns the number of context variables in the source."""
        # Each context var is repeated for each data var
        return len(self.context_vars) * self.n_data_variables()

    def get_output_variables_mask(self):
        """Returns a list M of length n_data_variables, where M[i] is True if the i-th
        variable is an output variable, and False if it is an input-only variable."""
        return self.output_vars_mask

    def __repr__(self):
        return f"Source(name={self.name}, \n\
                 dim={self.dim}, \n\
                 data vars={self.data_vars}, \n\
                 context vars={self.context_vars}, \n\
                 input-only vars={self.input_only_vars})"
