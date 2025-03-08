"""Implements the RMSE for multi-sources data."""

import torch


def multisource_rmse(y_pred, batch, avail_flags, **unused_kwargs):
    """Computes the RMSE between y_pred and y_true.
    Args:
        y_pred (dict): Dictionary {source: y_pred_s} where y_pred_s is a tensor of
            shape (B, C, ...) giving the predicted values for the source s.
        batch (dict of dict of torch.Tensor): Dict mapping source names to data dicts.
            For each source, batch[source] must contains the following entries: "values",
            "avail_mask".
        avail_flags (dict): Dictionary {source: avail_flag_s} where avail_flag_s is a
            tensor of shape (B,) containing 1 if the value is available, 0 if it was
            available and masked, and -1 if it was not available.
            The RMSE will only be computed for the samples and sources where the avail
            flag is 0.
    Returns:
        rmse (dict): Dictionary {source: rmse_s} where rmse_s is a scalar.
    """
    rmse = {}
    for source, y_pred_s in y_pred.items():
        y_true_s = batch[source]["values"]
        avail_flag_s = avail_flags[source] == 0
        avail_mask_s = batch[source]["avail_mask"].unsqueeze(1)  # (B, 1, ...)
        avail_mask_s = avail_mask_s[avail_flag_s] > -1  # True at every available point

        y_true_s = y_true_s[avail_flag_s]
        y_pred_s = y_pred_s[avail_flag_s]

        # If the avail_flag was never 0, we can skip the computation
        if y_true_s.numel() == 0:
            continue

        # Squared error
        se_s = torch.pow(y_pred_s - y_true_s, 2)  # (B, C, ...)
        se_s[~avail_mask_s] = 0 # Ignore the values that were unavailable
        # Compute the mean over each sample. We can't just call mean() as the
        # points set to zero would bias the result. We need to compute the sum
        # and divide by the number of non-masked values.
        non_batch_dims = tuple(range(1, se_s.dim()))
        mse_s = se_s.sum(dim=non_batch_dims) / avail_mask_s.sum(dim=non_batch_dims)

        # Deduce the RMSE of each sample and average it over the batch
        rmse_s = torch.sqrt(mse_s).mean()

        rmse[source] = rmse_s
    return rmse
