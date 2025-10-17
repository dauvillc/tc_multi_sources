"""Implements the QuantativeEvaluation class."""

from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
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
    - Skill-Spread Ratio (SSR). This is only computed if the model has multiple
        realizations per sample.
    The per-sample metrics are saved to disk in a JSON file. Then, the aggregated metrics
    are computed and saved to disk in a separate JSON file. Figures to compare the
    models are generated and saved to disk.
    """

    def __init__(self, model_data, parent_results_dir, num_workers=1, **kwargs):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
            num_workers (int): Number of workers for parallel processing.
            **kwargs: Additional keyword arguments passed to the AbstractMultisourceEvaluationMetric.
        """
        super().__init__(
            id_name="quantitative",
            full_name="Quantitative Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )
        self.num_workers = num_workers
        self.overall_metrics_dir = self.metric_results_dir / "overall"
        self.rmse_dir = self.metric_results_dir / "rmse"
        self.mae_dir = self.metric_results_dir / "mae"
        self.crps_dir = self.metric_results_dir / "crps"
        self.ssr_dir = self.metric_results_dir / "ssr"
        # Create the directories if they don't exist
        self.overall_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.rmse_dir.mkdir(parents=True, exist_ok=True)
        self.mae_dir.mkdir(parents=True, exist_ok=True)
        self.crps_dir.mkdir(parents=True, exist_ok=True)
        self.ssr_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, **kwargs):
        """Main evaluation method that processes the data for all models."""

        # Evaluate all models and save the results
        results = self._evaluate_models()
        results_file = self.metric_results_dir / "full_results.json"
        results.to_json(results_file, orient="records", lines=True)
        print(f"Full results saved to: {results_file}")

        # Compute and save the aggregated results
        self._save_aggregated_results(results)

        # Generate and save plots comparing the models
        self._plot_results(results)
        return

    def _evaluate_models(self):
        """Evaluates all models at once and returns the results.
        Returns:
            results (pd.DataFrame): DataFrame containing the evaluation results for the model.
                Includes the columns 'model_id', 'sample_index', 'source_name', 'source_index',
                'channel', 'mae', 'mse', 'crps', 'ssr'.
        """
        # Collect all samples into a list for parallel processing
        samples = list(self.samples_iterator())

        if self.num_workers > 1:
            # Parallel processing
            results = []
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(self._process_sample, sample_df, sample_data)
                    for sample_df, sample_data in samples
                ]
                # Process results with progress bar
                for future in tqdm(
                    as_completed(futures), desc="Evaluating samples", total=len(futures)
                ):
                    sample_results = future.result()
                    results.extend(sample_results)
        else:
            # Sequential processing
            results = []
            for sample_df, sample_data in tqdm(
                samples, desc="Evaluating samples", total=self.n_samples
            ):
                sample_results = self._process_sample(sample_df, sample_data)
                results.extend(sample_results)

        # Concatenate all results into a single DataFrame
        return pd.DataFrame(results)

    def _process_sample(self, sample_df, sample_data):
        """Process a single sample and return the results (helper method for parallel execution).

        Args:
            sample_df (pandas.DataFrame): DataFrame with sample metadata
            sample_data (dict): Dictionary containing targets and predictions

        Returns:
            list: List of result dictionaries for this sample
        """
        results = []
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
                    sample_results_dict = {
                        "model_id": model_id,
                        "sample_index": sample_index,
                        "source_name": source_name,
                        "source_index": source_index,
                        "channel": channel,
                        "mae": mae,
                        "mse": mse,
                        "crps": crps,
                    }
                    n_real = pred_data_channel.shape[0]
                    if n_real > 1:
                        # If there are multiple realizations, we can compute the SSR.
                        # We'll here just compute the MSE and the ensemble member variance,
                        # and aggregate them later to get the SSR.
                        ensemble_var, ensemble_mean_mse = self._compute_err_and_member_var(
                            pred_data_channel, target_data_channel
                        )
                        sample_results_dict["ssr"] = np.sqrt(((n_real + 1) / n_real)) * (
                            ensemble_var / ensemble_mean_mse
                        )
                    results.append(sample_results_dict)

        return results

    def _save_aggregated_results(self, results):
        """Computes and saves the aggregated results for all models.

        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results.
        """
        # Compute aggregated metrics for each model: mean and std.
        agg_results = (
            results.groupby("model_id")
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                mse_mean=("mse", "mean"),
                mse_std=("mse", "std"),
                crps_mean=("crps", "mean"),
                crps_std=("crps", "std"),
            )
            .reset_index()
        )

        # Compute bootstrap 95% confidence intervals for the mean metrics
        for model_id in agg_results["model_id"]:
            model_results = results[results["model_id"] == model_id]

            for metric in ["mae", "mse", "crps"]:
                data = model_results[metric].values
                # Bootstrap confidence interval
                res = stats.bootstrap(
                    (data,), np.mean, n_resamples=1000, confidence_level=0.95, method="percentile"
                )
                agg_results.loc[agg_results["model_id"] == model_id, f"{metric}_ci_lower"] = (
                    res.confidence_interval.low
                )
                agg_results.loc[agg_results["model_id"] == model_id, f"{metric}_ci_upper"] = (
                    res.confidence_interval.high
                )

        # Compute RMSE from MSE
        agg_results["rmse_mean"] = np.sqrt(agg_results["mse_mean"])
        agg_results["rmse_std"] = 0.5 * agg_results["mse_std"] / np.sqrt(agg_results["mse_mean"])

        # Bootstrap CI for RMSE
        for model_id in agg_results["model_id"]:
            model_results = results[results["model_id"] == model_id]
            data = model_results["mse"].values
            res = stats.bootstrap(
                (data,),
                lambda x: np.sqrt(np.mean(x)),
                n_resamples=1000,
                confidence_level=0.95,
                method="percentile",
            )
            agg_results.loc[agg_results["model_id"] == model_id, "rmse_ci_lower"] = (
                res.confidence_interval.low
            )
            agg_results.loc[agg_results["model_id"] == model_id, "rmse_ci_upper"] = (
                res.confidence_interval.high
            )

        # Save the aggregated results to a JSON file
        agg_results_file = self.metric_results_dir / "aggregated_results.json"
        agg_results.to_json(agg_results_file, orient="records", lines=True)
        print(f"Aggregated results saved to: {agg_results_file}")
        return

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
        overall_mae_plot_file = self.overall_metrics_dir / "mae_all_models.png"
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
        overall_rmse_plot_file = self.overall_metrics_dir / "rmse_all_models.png"
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
        overall_crps_plot_file = self.overall_metrics_dir / "crps_all_models.png"
        plt.savefig(overall_crps_plot_file)
        plt.close()
        print(f"Overall CRPS plot saved to: {overall_crps_plot_file}")

        # SSR: boxplot, only for models that have multiple realizations
        if "ssr" in results.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                x="model_id",
                y="ssr",
                data=results,
                showfliers=False,
            )
            plt.title("Skill-Spread Ratio for all models, sources and channels")
            plt.xlabel("Model ID")
            plt.ylabel("Skill-Spread Ratio (SSR)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            overall_ssr_plot_file = self.overall_metrics_dir / "ssr_all_models.png"
            plt.savefig(overall_ssr_plot_file)
            plt.close()
            print(f"Overall SSR plot saved to: {overall_ssr_plot_file}")

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
            mae_plot_file = self.mae_dir / f"mae_{source_name}_{channel}.png"
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
            rmse_plot_file = self.rmse_dir / f"rmse_{source_name}_{channel}.png"
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
            target_data (np.ndarray): Target data, of shape (...) matching the shape of each
                realization in pred_data.
        """
        pred_data, target_data = flatten_and_ignore_nans(pred_data, target_data)
        # Compute the first term: the mean absolute error between predictions and target
        term1 = np.abs(pred_data - target_data).mean()
        # Compute the second term: the mean absolute error between all pairs of predictions
        term2 = np.abs(pred_data[:, None] - pred_data[None, :]).mean()
        crps = term1 - 0.5 * term2
        return crps.item()

    @staticmethod
    def _compute_err_and_member_var(pred_data, target_data):
        """Computes the unbiased MSE between the ensemble mean and the target,
        as well as the ensemble member variance.

        Args:
            pred_data (np.ndarray): Predicted data, of shape (M, ...) where M is the number of
                realizations.
            target_data (np.ndarray): Target data, of shape (...) matching the shape of each
                realization in pred_data.
        Returns:
            var (float): Ensemble member variance.
            mean_mse (float): Debiased MSE between the ensemble mean and the target.
        """
        pred_data, target_data = flatten_and_ignore_nans(pred_data, target_data)
        K = pred_data.shape[0]  # Number of ensemble members
        # Compute the ensemble mean
        ensemble_mean = pred_data.mean(axis=0)
        # Compute the finite-sample variance of the ensemble members
        var = ((pred_data - ensemble_mean[None, :]) ** 2).mean() * (K / (K - 1))
        # Compute the unbiased mean squared error (MSE)
        mean_mse = ((ensemble_mean - target_data) ** 2).mean()
        return var, mean_mse
