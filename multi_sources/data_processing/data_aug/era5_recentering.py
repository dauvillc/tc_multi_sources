"""Implements the ERA5GridRecentering class."""

import torch


class ERA5GridRecentering:
    """Designed for samples where ERA5 is available in both the inputs,
    and as a forecast target. Regrids the target ERA5 patches so that they are
    centered on the same lat/lon as the latest input ERA5 patch.
    The recentering is done by taking the input patch's coordinates and copying
    the overlapping values from the forecast patch onto the new grid.

    Expects samples from the dataset as dicts {(source_name, index): data} where
    source_name is the name of the source and index is an integer representing
    the observation index (0 = most recent, 1 = second most recent, etc.).
    Each data dict contains the following key-value pairs:
    - "dt" is a scalar tensor of shape (1,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "characs" is a tensor of shape (n_charac_vars,) containing the source characteristics.
        Each data variable within a source has its own charac variables, which are all
        concatenated into a single tensor.
    - "coords" is a tensor of shape (2, H, W) containing the latitude and longitude at each pixel.
    - "landmask" is a tensor of shape (H, W) containing the land mask.
    - "dist_to_center" is a tensor of shape (H, W) containing the distance
        to the center of the storm.
    - "values" is a tensor of shape (n_variables, H, W) containing the variables for the source.
    """

    def __init__(self):
        pass

    def __call__(self, data):
        """Regrids the target ERA5 patches to a wider grid.
        Args:
            data (dict): Dict containing the data of all sources.
        Returns:
            dict: The data with the ERA5 target patches regridded.
        """
        # Find the latest input ERA5 patch, which is the first one with dt > 0.
        # For all indexes of ERA5 before that (dt <= 0, forecast patches), we will regrid.
        input_era5_key, forecast_keys = None, []
        for (source_name, index), sample in data.items():
            if source_name == "tc_primed_era5":
                if sample["dt"] > 0:
                    input_era5_key = (source_name, index)
                    break
                else:
                    forecast_keys.append((source_name, index))
        if input_era5_key is None:
            raise ValueError("No input ERA5 patch found in the data.")

        input_coords = data[input_era5_key]["coords"]  # Shape (2, H_in, W_in)
        _, H_in, W_in = input_coords.shape
        input_lat, input_lon = input_coords[0, :, 0], input_coords[1, 0]
        # Get the extent of the input patch's coordinates
        min_lat, max_lat = input_lat[-1], input_lat[0]
        min_lon = input_lon[0]
        # We'll compute the max_lon based on the width of the patch as it
        # generalizes to longitudes crossing the dateline.
        max_lon = min_lon + (W_in - 1) * 0.25

        # For all forecast ERA5 patches, regrid them to the input patch's grid.
        for frc_key in forecast_keys:
            forecast = data[frc_key]
            forecast_coords = forecast["coords"]  # Shape (2, H_frc, W_frc)
            _, H_frc, W_frc = forecast_coords.shape
            frc_lat, frc_lon = forecast_coords[0, :, 0], forecast_coords[1, 0]
            # Find the extent of the forecast patch's coordinates
            frc_min_lat, frc_max_lat = frc_lat[-1], frc_lat[0]
            frc_min_lon = frc_lon[0]
            # Similarly, compute the max_lon based on the width of the patch.
            frc_max_lon = frc_min_lon + (W_frc - 1) * 0.25
            # We can now compute where the forecast patch's grid falls
            # on the input patch's grid, given that both are on regular 0.25 deg grids.
            res = 0.25
            lat_start_idx = int(max((max_lat - frc_max_lat) / res, 0))
            lat_end_idx = int(min((H_in - (frc_min_lat - min_lat) / res), H_in))
            lon_start_idx = int(max((frc_min_lon - min_lon) / res, 0))
            lon_end_idx = int(min((W_in - (max_lon - frc_max_lon) / res), W_in))
            # We can also compute where the input patch's grid falls
            # on the forecast patch's grid.
            frc_lat_start_idx = int(max((frc_max_lat - max_lat) / res, 0))
            frc_lat_end_idx = int(min(H_frc - (min_lat - frc_min_lat) / res, H_frc))
            frc_lon_start_idx = int(max((min_lon - frc_min_lon) / res, 0))
            frc_lon_end_idx = int(min(W_frc - (frc_max_lon - max_lon) / res, W_frc))

            # We can now create the regridded forecast patch by, for each key,
            # creating an empty tensor and copying onto it the overlapping values.
            new_forecast = {}
            slice_lat = slice(lat_start_idx, lat_end_idx)
            slice_lon = slice(lon_start_idx, lon_end_idx)
            frc_slice_lat = slice(frc_lat_start_idx, frc_lat_end_idx)
            frc_slice_lon = slice(frc_lon_start_idx, frc_lon_end_idx)
            # - values (shape (n_variables, H_in, W_in))
            new_values = torch.full(
                (forecast["values"].shape[0], H_in, W_in),
                float("nan"),
            )
            new_values[:, slice_lat, slice_lon] = forecast["values"][
                :, frc_slice_lat, frc_slice_lon
            ]
            # - landmask and dist_to_center (shape (H_in, W_in))
            new_landmask = torch.full(
                (H_in, W_in),
                float("nan"),
            )
            new_landmask[slice_lat, slice_lon] = forecast["landmask"][frc_slice_lat, frc_slice_lon]
            new_dist_to_center = torch.full(
                (H_in, W_in),
                float("nan"),
            )
            new_dist_to_center[slice_lat, slice_lon] = forecast["dist_to_center"][
                frc_slice_lat, frc_slice_lon
            ]
            new_forecast["values"] = new_values
            new_forecast["landmask"] = new_landmask
            new_forecast["dist_to_center"] = new_dist_to_center
            # For the coords, use the input patch's coords
            new_forecast["coords"] = input_coords.clone()
            # Copy other keys without modification
            for data_key in ["dt", "characs"]:
                if data_key not in forecast:
                    continue
                new_forecast[data_key] = forecast[data_key].clone()
            # Update the data dict
            data[frc_key] = new_forecast
        return data
