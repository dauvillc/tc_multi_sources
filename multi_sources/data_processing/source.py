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
        data_vars (list of str): The names of the variables in the source, i.e.
            variables that can be found in a netCDF file of the source.
        charac_vars (dict of str -> dict of str -> float):
            Map {charac_var_name -> {data_var_name -> value}}.
            Those variables characterize the source within its source type, such as the
            observing frequency for a passive microwave source.
        input_only_vars (list of str, optional): The names of the variables that are input-only,
            which means they are not trained on. Defaults to [], i.e. all variables are both
            used as input and target.
        output_only_vars (list of str, optional): The names of the variables that are output-only,
            which means they are not used as input. Defaults to [], i.e. all variables are both
            used as input and target.
    """

    def __init__(
        self,
        source_name,
        source_type,
        dim,
        data_vars,
        charac_vars,
        input_only_vars=[],
        output_only_vars=[],
        **kwargs,
    ):
        self.name = source_name
        self.dim = dim
        self.type = source_type
        self.data_vars = data_vars
        self.input_vars = [var for var in data_vars if var not in output_only_vars]
        self.output_vars = [var for var in data_vars if var not in input_only_vars]
        # Make sure the input-only variables are in the data variables
        for var in input_only_vars:
            if var not in self.data_vars:
                raise ValueError(f"Input-only variable {var} not found in data variables.")
        # Make sure the output-only variables are in the data variables
        for var in output_only_vars:
            if var not in self.data_vars:
                raise ValueError(f"Output-only variable {var} not found in data variables.")
        # Pre-compute the input and output variables mask
        self.input_vars_mask = [var in self.input_vars for var in data_vars]
        self.output_vars_mask = [var in self.output_vars for var in data_vars]

        # Characteristic variables: only keep the entries that are in data_vars
        self.charac_vars = {}
        for charac_var_name, charac_vars in charac_vars.items():
            self.charac_vars[charac_var_name] = {
                data_var_name: value
                for data_var_name, value in charac_vars.items()
                if data_var_name in self.data_vars
            }
        # Pre-compute the list of the values of all charac variables
        self.charac_values = [value for _, _, value in self.iter_charac_variables()]

    def n_data_variables(self):
        """Returns the number of data variables in the source."""
        return len(self.data_vars)

    def n_input_variables(self):
        return len(self.input_vars)

    def n_target_variables(self):
        return len(self.output_vars)

    def n_charac_variables(self):
        """Returns the number of charac variables in the source,
        counting those of all data variables."""
        return sum([len(vars) for vars in self.charac_vars.values()])

    def iter_charac_variables(self):
        """Yields successive pairs (charac_var_name, data_var_name, value) where
        value is the value of the charac variable charac_var_name for the data variable
        data_var_name."""
        for charac_var_name, charac_vars in self.charac_vars.items():
            for data_var_name, value in charac_vars.items():
                yield charac_var_name, data_var_name, value

    def get_charac_values(self):
        """Returns the list of the values of all charac variables."""
        return self.charac_values

    def get_input_variables_mask(self):
        """Returns a list M of length n_data_variables, where M[i] is True if the i-th
        variable is an input variable, and False if it is an output-only variable."""
        return self.input_vars_mask

    def get_output_variables_mask(self):
        """Returns a list M of length n_data_variables, where M[i] is True if the i-th
        variable is an output variable, and False if it is an input-only variable."""
        return self.output_vars_mask

    def __repr__(self):
        return f"Source(name={self.name}, \n\
                 dim={self.dim}, \n\
                 data vars={self.data_vars}, \n\
                 charac vars={self.charac_vars}, \n\
                 input vars={self.input_vars}, \n\
                 output vars={self.output_vars}, \n\
                "
