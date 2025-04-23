import pandas as pd
import plotly.express as px
import xarray as xr

from .abstract_evaluation_metric import AbstractSourcewiseEvaluationMetric


class ScalarEvaluation(AbstractSourcewiseEvaluationMetric):
    """A metric that displays the distribution of each variable of a 0d (scalar)
    source.
    """

    def __init__(self, predictions_dir, results_dir, source_name, var_names, var_units):
        """Args:
        predictions_dir (Path): Directory with the predictions.
        results_dir (Path): Directory where the results will be saved.
        source_name (str): Name of the source to evaluate.
        var_names (list of str): List of length C where C is the number of variables
            (channels) in the source. Each element is the name of the variable.
        var_units (list of str): Same format as var_names, units of the variables.
        """
        super().__init__("scalar_eval", "Scalar evaluation", predictions_dir, results_dir)
        self.source_name = source_name
        self.var_names = var_names
        self.var_units = var_units

    def evaluate_source(self, source_name, info_df, **kwargs):
        """
        Processes a single source to compute scalar metrics.
        """
        # If the source does not match the one we want to evaluate, return
        if source_name != self.source_name:
            return
        results_dir = self.create_source_results_dir(source_name)
        # Get the unique batch indices for this source and load every batch
        unique_batch_indices = info_df["batch_idx"].unique()
        targets, preds = [], []
        for batch_idx in unique_batch_indices:
            t, p = self.load_batch(source_name, batch_idx)
            # Only consider batches where both targets and predictions are available
            if t is not None and p is not None:
                targets.append(t)
                preds.append(p)
        # Concatenate the datasets along the batch dimension
        targets = xr.concat(targets, dim="samples")
        preds = xr.concat(preds, dim="samples")
        # Convert them to pandas dataframes
        channels = list(targets.data_vars.keys())
        targets = targets.to_dataframe()[channels]
        preds = preds.to_dataframe()[channels]
        # Display the distributions
        display_scalar_distributions(targets, preds, results_dir)


def display_scalar_distributions(targets_df, preds_df, results_dir):
    """Displays separate histograms for each column in the given
    target and prediction DataFrames, for side-by-side comparison.

    Args:
        targets_df (pd.DataFrame): DataFrame containing target values.
            Must contain the same columns as preds_df.
        preds_df (pd.DataFrame): DataFrame containing prediction values.
        results_dir (Path): Directory where the results will be saved.
    """
    for col in targets_df.columns:
        combined = pd.concat(
            [
                pd.DataFrame({"value": targets_df[col], "type": "Targets"}),
                pd.DataFrame({"value": preds_df[col], "type": "Predictions"}),
            ],
            ignore_index=True,
        )
        fig = px.histogram(
            combined, x="value", color="type", barmode="overlay", title=f"Distribution of {col}"
        )
        fig.add_vline(
            x=targets_df[col].mean(),
            line_dash="dash",
            line_color="blue",
            annotation_text="Groundtruths mean",
            yref="paper",
            y0=0,
            y1=1,
        )
        fig.add_vline(
            x=preds_df[col].mean(),
            line_dash="dash",
            line_color="orange",
            annotation_text="Predictions mean",
            yref="paper",
            y0=0,
            y1=1,
        )
        fig.write_html(results_dir / f"distribution_{col}.html")
