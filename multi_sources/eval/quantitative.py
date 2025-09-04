"""Implements the QuantativeEvaluation class."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


def flatten_and_ignore_nans(pred, target):
    """Given an array of predictions and targets, flattens them along all dimensions
    except the first one (realizations), and ignores NaNs in the target array."""
    # Create a mask of valid (non-NaN) entries in the target array
    valid_mask = ~np.isnan(target)
    # Apply the mask to both arrays and flatten them
    pred_flat = pred[:, valid_mask].reshape(pred.shape[0], -1)
    target_flat = target[valid_mask].flatten()
    return pred_flat, target_flat


class QuantitativeEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluation class that computes sample-wise metrics for a given set of models:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Continuous Ranked Probability Score (CRPS). For models with a single prediction,
        this is equivalent to the MAE.
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
                'channel', 'mae', 'mse', 'crps'.
        """
        results = []  # List of dictionaries that we'll concatenate later into a DataFrame
        for sample_df, sample_data in tqdm(
            self.samples_iterator(), desc="Evaluating samples", total=self.n_samples
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
                        pred_data_channel = pred_data[channel].values
                        target_data_channel = target_data[channel].values
                        # If there isn't a realization dimension, add one for consistency
                        if pred_data_channel.ndim == target_data_channel.ndim:
                            pred_data_channel = np.expand_dims(pred_data_channel, axis=0)
                        # Compute the metrics for the current channel
                        mae = self._compute_mae(pred_data_channel, target_data_channel)
                        mse = self._compute_mse(pred_data_channel, target_data_channel)
                        crps = self._compute_crps(pred_data_channel, target_data_channel)
                        results.append(
                            {
                                "model_id": model_id,
                                "sample_index": sample_index,
                                "source_name": source_name,
                                "source_index": source_index,
                                "channel": channel,
                                "mae": mae,
                                "mse": mse,
                                "crps": crps,
                            }
                        )

        # Concatenate all results into a single DataFrame
        return pd.DataFrame(results)

    def _plot_results(self, results):
        """Generates and saves plots comparing the models based on the evaluation results.
        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results.
        """
        sns.set_theme(style="whitegrid")
        # First, we'll show plots of the metrics for each model, over all sources and channels.
        # This gives a general overview of the models' performance.
        # MAE: we'll make a boxplot.
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x="model_id",
            y="mae",
            data=results,
            showfliers=False,
        )
        plt.title("MAE for all models, sources and channels")
        plt.xlabel("Model ID")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        overall_mae_plot_file = self.metric_results_dir / "mae_all_models.png"
        plt.savefig(overall_mae_plot_file)
        plt.close()
        print(f"Overall MAE plot saved to: {overall_mae_plot_file}")

        # RMSE: we'll make a barplot, since the RMSE is a single value per model.
        rmse_per_model = results.groupby("model_id")["mse"].mean().reset_index()
        rmse_per_model["rmse"] = np.sqrt(rmse_per_model["mse"])
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model_id",
            y="rmse",
            data=rmse_per_model,
        )
        plt.title("RMSE for all models, sources and channels")
        plt.xlabel("Model ID")
        plt.ylabel("Root Mean Squared Error (RMSE)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        overall_rmse_plot_file = self.metric_results_dir / "rmse_all_models.png"
        plt.savefig(overall_rmse_plot_file)
        plt.close()
        print(f"Overall RMSE plot saved to: {overall_rmse_plot_file}")

        # CRPS: boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x="model_id",
            y="crps",
            data=results,
            showfliers=False,
        )
        plt.title("CRPS for all models, sources and channels")
        plt.xlabel("Model ID")
        plt.ylabel("Continuous Ranked Probability Score (CRPS)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        overall_crps_plot_file = self.metric_results_dir / "crps_all_models.png"
        plt.savefig(overall_crps_plot_file)
        plt.close()
        print(f"Overall CRPS plot saved to: {overall_crps_plot_file}")

        # Now, we'll separate the plots by pair (source_name, channel). In each plot,
        # the x-axis will be the model_id and the y-axis will be the metric.
        grouped_results = results.groupby(["source_name", "channel"])
        for (source_name, channel), group in grouped_results:
            # MAE plot
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

            # RMSE
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

    @staticmethod
    def _compute_mae(pred_data, target_data):
        """Computes the Mean Absolute Error (MAE) between predictions and targets."""
        pred_flat, target_flat = flatten_and_ignore_nans(pred_data, target_data)
        return np.abs((pred_flat - target_flat)).mean().item()

    @staticmethod
    def _compute_mse(pred_data, target_data):
        """Computes the Mean Squared Error (MSE) between predictions and targets."""
        pred_flat, target_flat = flatten_and_ignore_nans(pred_data, target_data)
        return ((pred_flat - target_flat) ** 2).mean().item()

    @staticmethod
    def _compute_crps(pred_data, target_data):
        """Computes the Continuous Ranked Probability Score (CRPS) between predictions and targets.
        For deterministic predictions, this is equivalent to the MAE.

        Args:
            pred_data (np.ndarray): Predicted data, of shape (M, ...) where M is the number of
                realizations, or (...) for deterministic predictions.
        """
        pred_data, target_data = flatten_and_ignore_nans(pred_data, target_data)
        # Compute the first term: the mean absolute error between predictions and target
        term1 = np.abs(pred_data - target_data).mean()
        # Compute the second term: the mean absolute error between all pairs of predictions
        term2 = np.abs(pred_data[:, None] - pred_data[None, :]).mean()
        crps = term1 - 0.5 * term2
        return crps.item()
