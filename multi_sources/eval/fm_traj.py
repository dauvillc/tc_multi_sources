"""Implements the FlowMatchingTrajectoryEvaluation class."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


class FlowMatchingTrajectoryEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluates the errors of the predicted velocity fields at each integration
    step, to assess where the error accumulates along the trajectory.
    """

    def __init__(self, model_data, parent_results_dir, **kwargs):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
            **kwargs: Additional keyword arguments passed to the AbstractMultisourceEvaluationMetric.
        """
        super().__init__(
            id_name="flow_matching_trajectory",
            full_name="Flow Matching Trajectory Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
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
                Includes the following columns:
                - model_id: Identifier of the model.
                - sample_index: Index of the sample.
                - source: Name of the reconstructed source.
                - channel: Name of the reconstructed channel.
                - integration_step: Index of the integration step.
                - vf_mse: Mean Squared Error of the velocity field at this step.

        """
        results = []  # List of dictionaries that we'll concatenate later into a DataFrame
        for sample_df, sample_data in tqdm(
            self.samples_iterator(include_intermediate_steps=True),
            desc="Evaluating samples",
            total=self.n_samples,
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
                    true_vf_data = sample_data["true_vf"][model_id][src]
                    time_grid = true_vf_data["integration_step"].values
                    for channel in channels:
                        pred_data_channel = pred_data[channel].values
                        true_vf_data_channel = true_vf_data[channel].values
                        # At this point, the targets have shape (...) and the predictions
                        # (R, T, ...) where R is the number of realizations and T the number of
                        # integration steps.
                        vf_mse = self._compute_vf_mse(
                            pred_data_channel, true_vf_data_channel, time_grid
                        )  # Shape (T,)
                        for step, mse in zip(range(len(time_grid)), vf_mse):
                            results.append(
                                {
                                    "model_id": model_id,
                                    "sample_index": sample_index,
                                    "source": source_name,
                                    "source_index": source_index,
                                    "channel": channel,
                                    "integration_step": step,
                                    "vf_mse": mse,
                                }
                            )
        results = pd.DataFrame(results)
        return results

    @staticmethod
    def _compute_vf_mse(pred, true_vf, time_grid):
        """Computes the Mean Squared Error (MSE) between the predicted and target
        velocity fields, ignoring NaNs in the target array.

        Args:
            pred (np.ndarray): Predicted velocity field of shape (R, T, ...).
            true_vf (np.ndarray): Target velocity field of shape (R, T, ...).
            time_grid (np.ndarray): Time grid of shape (T,) used for integration.
        Returns:
            vf_mse (np.ndarray): MSE of shape (T,) for each integration step.
        """
        # We first need to compute the velocity fields from the integrated positions
        # by taking the difference between consecutive steps
        vf_pred = np.diff(pred, axis=1)
        # Compute the MSE for each integration step, ignoring NaNs
        vf_mse = (vf_pred - true_vf) ** 2
        vf_mse = vf_mse.reshape(vf_mse.shape[0], vf_mse.shape[1], -1)
        vf_mse = np.nanmean(vf_mse, axis=(0, 2))  # Mean over realizations and spatial dims
        return vf_mse

    def _plot_results(self, results):
        """Generates and saves plots comparing the models based on the evaluation results.
        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results for the model.
        """
        # Plot settings
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 8))

        # First, we'll plot the average velocity field MSE over all samples,
        # for each model, over all channels and sources.
        fig, axes = plt.subplots(figsize=(12, 8))
        sns.lineplot(
            data=results,
            x="integration_step",
            y="vf_mse",
            hue="model_id",
            errorbar="sd",
            ax=axes,
        )
        axes.set_title("Average Velocity Field MSE over Integration Steps")
        axes.set_xlabel("Integration Step")
        axes.set_ylabel("Velocity Field MSE")
        axes.legend(title="Model ID")
        plot_file = self.metric_results_dir / "avg_vf_mse_over_steps.png"
        plt.savefig(plot_file)
        plt.close(fig)
        print(f"Plot saved to: {plot_file}")

        # Next, we'll plot the average velocity field MSE over all samples,
        # all sources, but separately for each channel.
        channels = results["channel"].unique()
        for channel in channels:
            fig, axes = plt.subplots(figsize=(12, 8))
            channel_data = results[results["channel"] == channel]
            sns.lineplot(
                data=channel_data,
                x="integration_step",
                y="vf_mse",
                hue="model_id",
                errorbar="sd",
                ax=axes,
            )
            axes.set_title(
                f"Average Velocity Field MSE over Integration Steps - Channel: {channel}"
            )
            axes.set_xlabel("Integration Step")
            axes.set_ylabel("Velocity Field MSE")
            axes.legend(title="Model ID")
            plot_file = self.metric_results_dir / f"avg_vf_mse_over_steps_channel_{channel}.png"
            plt.savefig(plot_file)
            plt.close(fig)
            print(f"Plot saved to: {plot_file}")

        # Finally, we'll plot the average velocity field MSE over all samples,
        # all channels, but separately for each source.
        sources = results["source"].unique()
        for source in sources:
            fig, axes = plt.subplots(figsize=(12, 8))
            source_data = results[results["source"] == source]
            sns.lineplot(
                data=source_data,
                x="integration_step",
                y="vf_mse",
                hue="model_id",
                errorbar="sd",
                ax=axes,
            )
            axes.set_title(f"Average Velocity Field MSE over Integration Steps - Source: {source}")
            axes.set_xlabel("Integration Step")
            axes.set_ylabel("Velocity Field MSE")
            axes.legend(title="Model ID")
            plot_file = self.metric_results_dir / f"avg_vf_mse_over_steps_source_{source}.png"
            plt.savefig(plot_file)
            plt.close(fig)
            print(f"Plot saved to: {plot_file}")
