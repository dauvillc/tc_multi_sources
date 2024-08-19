"""Implements small utility functions for data processing."""


def _get_leaf_subsources(source_dict, path="", previous_vars=[]):
    """Returns the leaf subsources of a source dictionary."""
    # Recursivity stop condition: if no subsource key is found, return the source
    # with its varables as well as the previous ones. Replace
    # the dim key if it is already present.
    subsource_keys = [key for key in source_dict.keys() if key != "variables"]
    if not subsource_keys:
        return {path: (previous_vars + source_dict.get("variables", []))}
    # If there are subsource keys, call the function recursively on each subsource.
    returned_dict = {}
    for subsource_key in subsource_keys:
        returned_dict.update(
            _get_leaf_subsources(
                source_dict[subsource_key],
                path + "_" + subsource_key,  # source_subsource_ ... _lastsubsource
                previous_vars + source_dict.get("variables", []),
            )
        )
    return returned_dict


def read_variables_dict(variables_dict):
    """Reads the variables dictionary that specifies which variables should be
    included from which source.
    Args:
        variables_dict: dictionary with the following structure:
            {
                "source1": {
                    "subsource1": {
                        "variables": ["var1", "var2", ...],
                        "subsource2": {
                            "variables": ["var3", "var4", ...],
                            ...
                        },
                        ...
                    },
                    ...
                },
                "source2": {
                    ...
                },
                ...
            }
    Returns:
        A dictionary with the following structure:
            {
                "source1_subsource1_subsource2_..._lastsubsource": ["var1", "var2", ...],
                ...
            }
    """
    result = _get_leaf_subsources(variables_dict)
    # Remove the initial '_' at the beginning of each source key.
    return {key[1:]: value for key, value in result.items()}
