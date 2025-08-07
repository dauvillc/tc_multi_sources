"""Implements the QuantativeEvaluation class."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


class QuantitativeEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluation class that computes sample-wise metrics for a given set of models:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    The per-sample metrics are saved to disk in a JSON file. Then, the aggregated metrics
    are computed and saved to disk in a separate JSON file. Figures to compare the
    models are generated and saved to disk.
    """

    def __init__(self, model_data, parent_results_dir):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
        """
        super().__init__(
            id_name="quantitative",
            full_name="Quantitative Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
        )

    def evaluate(self, **kwargs):
        """Main evaluation method that processes the data for all models."""

        # Evaluate all models and save the results
        results = self._evaluate_models()
        results_file = self.metric_results_dir / "full_results.json"
        results.to_json(results_file, orient="records", lines=True)
        print(f"Full results saved to: {results_file}")

        # Generate and save plots comparing the models
        self._plot_results(results)
        return

    def _evaluate_models(self):
        """Evaluates all models at once and returns the results.
        Returns:
            results (pd.DataFrame): DataFrame containing the evaluation results for the model.
                Includes the columns 'model_id', 'sample_index', 'source_name', 'source_index',
                'channel', 'mae', 'mse'.
        """
        results = []  # List of dictionaries that we'll concatenate later into a DataFrame
        for sample_df, sample_data in tqdm(
            self.samples_iterator(), desc="Evaluating samples", total=len(self.samples_df)
        ):
            sample_index = sample_df["sample_index"].iloc[0]
            for src, target_data in sample_data["targets"].items():
                source_name, source_index = src
                # We only evaluate sources that were masked, i.e. for which the availability flag
                # is 0 (1 meaning available but not masked, -1 meaning not available).
                if sample_df.loc[src, "avail"] != 0:
                    continue
                # Retrieve the list of channels for the source
                channels = list(target_data.data_vars.keys())
                # Evaluate each model's predictions against the target data
                # on every channel.
                for model_id in self.model_data:
                    pred_data = sample_data["predictions"][model_id][src]
                    for channel in channels:
                        # Compute the MAE and MSE for the current channel
                        mae = self._compute_mae(pred_data[channel], target_data[channel])
                        mse = self._compute_mse(pred_data[channel], target_data[channel])
                        results.append(
                            {
                                "model_id": model_id,
                                "sample_index": sample_index,
                                "source_name": source_name,
                                "source_index": source_index,
                                "channel": channel,
                                "mae": mae,
                                "mse": mse,
                            }
                        )

        # Concatenate all results into a single DataFrame
        return pd.DataFrame(results)

    @staticmethod
    def _compute_mae(pred_data, target_data):
        """Computes the Mean Absolute Error (MAE) between predictions and targets."""
        return np.abs((pred_data - target_data)).mean().item()

    @staticmethod
    def _compute_mse(pred_data, target_data):
        """Computes the Mean Squared Error (MSE) between predictions and targets."""
        return ((pred_data - target_data) ** 2).mean().item()

    def _plot_results(self, results):
        """Generates and saves plots comparing the models based on the evaluation results.
        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results.
        """
        sns.set_theme(style="whitegrid")
        # We'll separate the plots by pair (source_name, channel). In each plot,
        # the x-axis will be the model_id and the y-axis will be the metric (MAE or MSE).
        grouped_results = results.groupby(["source_name", "channel"])
        for (source_name, channel), group in grouped_results:
            # MAE plot: we'll make a boxplot.
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                x="model_id",
                y="mae",
                data=group,
                showfliers=False,
            )
            plt.title(f"MAE for {source_name} - {channel}")
            plt.xlabel("Model ID")
            plt.ylabel("Mean Absolute Error (MAE)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            mae_plot_file = self.metric_results_dir / f"mae_{source_name}_{channel}.png"
            plt.savefig(mae_plot_file)
            plt.close()
            print(f"MAE plot saved to: {mae_plot_file}")

            # RMSE plot: we won't plot directly the MSE but rather the RMSE (which is therefore
            # a single value per model instead of a distribution).
            rmse_per_model = group.groupby("model_id")["mse"].mean().reset_index()
            rmse_per_model["rmse"] = np.sqrt(rmse_per_model["mse"])
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x="model_id",
                y="rmse",
                data=rmse_per_model,
            )
            plt.title(f"RMSE for {source_name} - {channel}")
            plt.xlabel("Model ID")
            plt.ylabel("Root Mean Squared Error (RMSE)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            rmse_plot_file = self.metric_results_dir / f"rmse_{source_name}_{channel}.png"
            plt.savefig(rmse_plot_file)
            plt.close()
            print(f"RMSE plot saved to: {rmse_plot_file}")
