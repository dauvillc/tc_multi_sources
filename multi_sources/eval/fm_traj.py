"""Implements the FlowMatchingTrajectoryEvaluation class."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
from tqdm import tqdm

from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


class FlowMatchingTrajectoryEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluates the errors of the predicted velocity fields at each integration
    step, to assess where the error accumulates along the trajectory.
    """

    def __init__(self, model_data, parent_results_dir, visualize_trajectories=None, **kwargs):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
            visualize_trajectories (bool or float, optional): Whether to create plots that display the trajectory
                for a fraction of the samples. If a float between 0 and 1, indicates the fraction of samples
                to visualize. If True, visualizes all samples.
            **kwargs: Additional keyword arguments passed to the AbstractMultisourceEvaluationMetric.
        """
        super().__init__(
            id_name="flow_matching_trajectory",
            full_name="Flow Matching Trajectory Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
            **kwargs,
        )
        self.visualize_trajectories = visualize_trajectories

    def evaluate(self, **kwargs):
        """Main evaluation method that processes the data for all models."""

        # Evaluate all models and save the results
        results = self._evaluate_models()
        results_file = self.metric_results_dir / "full_results.json"
        results.to_json(results_file, orient="records", lines=True)
        print(f"Full results saved to: {results_file}")

        # Generate and save plots comparing the models
        self._plot_results(results)

        # Optionally display the trajectories for a fraction of the samples
        self._display_trajectories()
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
                                    "integration_step": time_grid[step],
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
        figs_dir = self.metric_results_dir / "vf_error"
        figs_dir.mkdir(exist_ok=True, parents=True)
        print(f"Writing plots to: {figs_dir}")

        # First, we'll plot the average velocity field MSE over all samples,
        # for each model, over all channels and sources.
        fig, axes = plt.subplots(figsize=(12, 8))
        sns.lineplot(
            data=results,
            x="integration_step",
            y="vf_mse",
            hue="model_id",
            markers="o",
            errorbar="sd",
            ax=axes,
        )
        axes.set_title("Average Velocity Field MSE over Integration Steps")
        axes.set_xlabel("Integration Step")
        axes.set_ylabel("Velocity Field MSE")
        axes.legend(title="Model ID")
        plot_file = figs_dir / "avg_vf_mse_over_steps.png"
        plt.savefig(plot_file)
        plt.close(fig)

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
                markers="o",
                errorbar="sd",
                ax=axes,
            )
            axes.set_title(
                f"Average Velocity Field MSE over Integration Steps - Channel: {channel}"
            )
            axes.set_xlabel("Integration Step")
            axes.set_ylabel("Velocity Field MSE")
            axes.legend(title="Model ID")
            plot_file = figs_dir / f"avg_vf_mse_over_steps_channel_{channel}.png"
            plt.savefig(plot_file)
            plt.close(fig)

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
                markers="o",
                errorbar="sd",
                ax=axes,
            )
            axes.set_title(f"Average Velocity Field MSE over Integration Steps - Source: {source}")
            axes.set_xlabel("Integration Step")
            axes.set_ylabel("Velocity Field MSE")
            axes.legend(title="Model ID")
            plot_file = figs_dir / f"avg_vf_mse_over_steps_source_{source}.png"
            plt.savefig(plot_file)
            plt.close(fig)

    def _display_trajectories(self):
        """Displays the trajectories for a fraction of the samples, if specified.
        For now, assumes the data is 2D (images), so the targets for a single channel
        has shape (H, W).
        """
        if self.visualize_trajectories is None:
            return
        elif isinstance(self.visualize_trajectories, bool):
            frac = 1.0
        elif (
            isinstance(self.visualize_trajectories, float) and 0 < self.visualize_trajectories < 1
        ):
            frac = self.visualize_trajectories
        max_samples = int(self.n_samples * frac)
        figs_dir = self.metric_results_dir / "displayed_trajectories"
        figs_dir.mkdir(exist_ok=True, parents=True)
        print(f"Writing trajectory plots to: {figs_dir}")

        import plotly.graph_objects as go

        from multi_sources.data_processing.grid_functions import crop_nan_border_numpy

        for k, (sample_df, sample_data) in tqdm(
            enumerate(self.samples_iterator(include_intermediate_steps=True)),
            desc="Creating trajectory animations",
            total=max_samples,
        ):
            sample_index = sample_df["sample_index"].iloc[0]
            # Retrieve the number of channels from any source. We'll assume
            # all sources have the same number of channels.
            channels = list(next(iter(sample_data["targets"].values())).data_vars)

            # We'll now create a plotly animation for each channel. Each animation will have
            # one column per source and one row per model. Each figure will use frames to show
            # the integration steps of the corresponding source and model.
            # Note: we'll always plot the target source on the first column, and we actually
            # expect the other sources to be static.
            # Note 2: we'll only plot the first realization, for simplicity.
            for channel_idx, channel in enumerate(channels):
                # Count the number of available or target sources (avail flag 0 or 1)
                # to determine the number of columns
                target_sources = [
                    src for src in sample_df.index if sample_df.loc[src, "avail"] == 0
                ]
                available_sources = [
                    src for src in sample_df.index if sample_df.loc[src, "avail"] == 1
                ]
                n_cols = len(target_sources) + len(available_sources)
                n_rows = len(self.model_data)

                # Create subplot titles
                subplot_titles = []
                for src in target_sources:
                    subplot_titles.append(f"Target: {src[0]}")
                for src in available_sources:
                    subplot_titles.append(f"Available: {src[0]}")

                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.05,
                )

                # Get the time steps for animation frames
                first_target = target_sources[0] if target_sources else None
                if (
                    first_target
                    and first_target in sample_data["predictions"][list(self.model_data.keys())[0]]
                ):
                    time_steps = sample_data["predictions"][list(self.model_data.keys())[0]][
                        first_target
                    ]["integration_step"].values
                else:
                    time_steps = [0]  # Fallback if no target sources with predictions

                # Prepare data for each model and source
                cropped_data = {}
                norms = {}  # Store normalization info for consistent color scales

                for model_idx, model_id in enumerate(self.model_data):
                    row = model_idx + 1
                    col = 1

                    # Process target sources first
                    for src in target_sources:
                        if src in sample_data["predictions"][model_id]:
                            # Get target data for reference
                            target_data = sample_data["targets"][src][channel].values
                            pred_data = sample_data["predictions"][model_id][src][channel].values

                            # Use first realization only
                            if len(pred_data.shape) > len(target_data.shape):
                                pred_data = pred_data[0]  # Shape: (T, H, W)

                            # Crop borders using target as reference
                            cropped_arrays = crop_nan_border_numpy(
                                target_data, [target_data, pred_data]
                            )
                            cropped_target = cropped_arrays[0]
                            cropped_pred = cropped_arrays[1]

                            # Store normalization based on target data
                            if src not in norms:
                                vmin, vmax = np.nanmin(cropped_target), np.nanmax(cropped_target)
                                norms[src] = (vmin, vmax)

                            cropped_data[(model_id, src)] = cropped_pred

                            # Add initial frame (will be updated in animation frames)
                            initial_frame = (
                                cropped_pred[0] if len(cropped_pred.shape) == 3 else cropped_pred
                            )
                            fig.add_trace(
                                go.Heatmap(
                                    z=initial_frame,
                                    colorscale="viridis",
                                    zmin=norms[src][0],
                                    zmax=norms[src][1],
                                    showscale=True if col == n_cols else False,
                                ),
                                row=row,
                                col=col,
                            )
                        col += 1

                    # Process available sources (these should be static)
                    for src in available_sources:
                        if src in sample_data["targets"]:
                            available_data = sample_data["targets"][src][channel].values

                            # Crop borders
                            cropped_available = crop_nan_border_numpy(
                                available_data, [available_data]
                            )[0]

                            # Store normalization
                            if src not in norms:
                                vmin, vmax = np.nanmin(cropped_available), np.nanmax(
                                    cropped_available
                                )
                                norms[src] = (vmin, vmax)

                            fig.add_trace(
                                go.Heatmap(
                                    z=cropped_available,
                                    colorscale="viridis",
                                    zmin=norms[src][0],
                                    zmax=norms[src][1],
                                    showscale=True if col == n_cols else False,
                                ),
                                row=row,
                                col=col,
                            )
                        col += 1

                # Create animation frames
                frames = []
                for t_idx, t in enumerate(time_steps):
                    frame_data = []

                    for model_idx, model_id in enumerate(self.model_data):
                        col = 1

                        # Target sources (animated)
                        for src in target_sources:
                            if (model_id, src) in cropped_data:
                                pred_data = cropped_data[(model_id, src)]
                                if len(pred_data.shape) == 3:  # Has time dimension
                                    frame_z = pred_data[t_idx]
                                else:
                                    frame_z = pred_data  # Static data

                                frame_data.append(
                                    go.Heatmap(
                                        z=frame_z,
                                        colorscale="viridis",
                                        zmin=norms[src][0],
                                        zmax=norms[src][1],
                                        showscale=True if col == n_cols else False,
                                    )
                                )
                            col += 1

                        # Available sources (static, same data for all frames)
                        for src in available_sources:
                            if src in sample_data["targets"]:
                                available_data = sample_data["targets"][src][channel].values
                                cropped_available = crop_nan_border_numpy(
                                    available_data, [available_data]
                                )[0]

                                frame_data.append(
                                    go.Heatmap(
                                        z=cropped_available,
                                        colorscale="viridis",
                                        zmin=norms[src][0],
                                        zmax=norms[src][1],
                                        showscale=True if col == n_cols else False,
                                    )
                                )
                            col += 1

                    frames.append(go.Frame(data=frame_data, name=f"Step {t}"))

                fig.frames = frames

                # Add animation controls
                fig.update_layout(
                    title=f"Sample {sample_index} - {channel} - Flow Matching Trajectory",
                    updatemenus=[
                        {
                            "buttons": [
                                {
                                    "args": [
                                        None,
                                        {
                                            "frame": {"duration": 500, "redraw": True},
                                            "fromcurrent": True,
                                        },
                                    ],
                                    "label": "Play",
                                    "method": "animate",
                                },
                                {
                                    "args": [
                                        [None],
                                        {
                                            "frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                    "label": "Pause",
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 10, "t": 87},
                            "showactive": False,
                            "type": "buttons",
                            "x": 0.1,
                            "xanchor": "right",
                            "y": 0,
                            "yanchor": "top",
                        }
                    ],
                    sliders=[
                        {
                            "active": 0,
                            "yanchor": "top",
                            "xanchor": "left",
                            "currentvalue": {
                                "font": {"size": 20},
                                "prefix": "Integration Step:",
                                "visible": True,
                                "xanchor": "right",
                            },
                            "transition": {"duration": 300, "easing": "cubic-in-out"},
                            "pad": {"b": 10, "t": 50},
                            "len": 0.9,
                            "x": 0.1,
                            "y": 0,
                            "steps": [
                                {
                                    "args": [
                                        [f"Step {t}"],
                                        {
                                            "frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 300},
                                        },
                                    ],
                                    "label": f"Step {t}",
                                    "method": "animate",
                                }
                                for t in time_steps
                            ],
                        }
                    ],
                )

                # Update layout for better appearance and maintain aspect ratio
                fig.update_layout(height=600 * n_rows, width=400 * n_cols, showlegend=False)

                # Set equal aspect ratio for all subplots
                for row in range(1, n_rows + 1):
                    for col in range(1, n_cols + 1):
                        fig.update_xaxes(scaleanchor=f"y{row}", scaleratio=1, row=row, col=col)
                        fig.update_yaxes(constrain="domain", row=row, col=col)

                plot_file = figs_dir / f"sample_{sample_index}_channel_{channel}.html"
                fig.write_html(plot_file)

            if k + 1 >= max_samples:
                break
