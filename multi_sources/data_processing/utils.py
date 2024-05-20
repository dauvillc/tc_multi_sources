"""Implements small utility functions for data processing."""

from multi_sources.data_processing.source import Source


def _get_leaf_subsources(source_dict, path="", previous_vars=[], previous_env_vars=[], dim=0):
    """Returns the leaf subsources of a source dictionary."""
    # Recursivity stop condition: if no subsource key is found, return the source
    # with its vars, env_vars, dim, as well as the previous vars and env_vars. Replace
    # the dim key if it is already present.
    subsource_keys = [
        key
        for key in source_dict.keys()
        if not key in ["variables", "environment_variables", "n_dimensions"]
    ]
    if not subsource_keys:
        return {
            path: (
                previous_vars + source_dict.get("variables", []),
                previous_env_vars + source_dict.get("environment_variables", []),
                source_dict.get("n_dimensions", dim),
            )
        }
    # If there are subsource keys, call the function recursively on each subsource.
    returned_dict = {}
    for subsource_key in subsource_keys:
        returned_dict.update(
            _get_leaf_subsources(
                source_dict[subsource_key],
                path + "." + subsource_key,  # source.subsource. ... .lastsubsource
                previous_vars + source_dict.get("variables", []),
                previous_env_vars + source_dict.get("environment_variables", []),
                source_dict.get("n_dimensions", dim),
            )
        )
    return returned_dict


def read_sources(sources_dict):
    """Reads the source dictionary and returns the content as a dictionary.

    Args:
        sources_dict (:obj:`dict`): Sources configuration dictionary.

    Returns:
        sources (:obj:`list` of :obj:`multi_sources.data_processing.source.Source`): List of sources.
    """
    # The following function will return a dictionary with the following structure:
    # {source_subsource. ... _lastsubsource: vars, env_vars, dim}
    sources = _get_leaf_subsources(sources_dict)
    # Rename the starting dot at the start of the source names
    sources = {key[1:]: value for key, value in sources.items()}
    # Create a list of Source objects from the dictionary
    source_list = []
    for name, (vars, env_vars, dim) in sources.items():
        source_list.append(Source(name, dim, vars, env_vars))
    return source_list
