"""Implements the RMSE for multi-sources data."""

import torch


def multisource_rmse(y_pred, y_true, masks, **unused_kwargs):
    """Computes the RMSE between y_pred and y_true.
    Args:
        y_pred (dict): Dictionary {source: y_pred_s} where y_pred_s is a tensor of
            shape (B, C, ...) giving the predicted values for the source s.
        y_true (dict): Dictionary {source: y_true_s} where y_true_s is a tensor of
            shape (B, C, ...) giving the true values for the source s.
        masks (dict): Dictionary {source: mask_s} where mask_s is a tensor of
            shape (B, ...), valued 1 at points that should be considered and 0
            at points that should be ignored.
    Returns:
        rmse (dict): Dictionary {source: rmse_s} where rmse_s is a scalar.
    """
    rmse = {}
    for source, y_pred_s in y_pred.items():
        y_true_s = y_true[source]
        mask_s = masks[source].unsqueeze(1).expand_as(y_pred_s)
        # Compute the RMSE
        rmse_s = torch.sqrt(torch.sum(mask_s * (y_pred_s - y_true_s) ** 2) / torch.sum(mask_s))
        rmse[source] = rmse_s
    return rmse
