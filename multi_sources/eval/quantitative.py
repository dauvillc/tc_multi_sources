"""
Implements the QuantitativeEvaluation class.
"""

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric

# Set publication-quality style settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",  # PDF for vector graphics in publications
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


class QuantitativeEvaluation(AbstractMultisourceEvaluationMetric):
    """Computes MSE and MAE between targets and predictions for all common channels.
    Works with any channel shape and dimension count.
    """

    def __init__(self, model_data, parent_results_dir, create_source_plots=True):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to dictionaries containing:
                - info_df: DataFrame with metadata
                - root_dir: Path to predictions directory
                - results_dir: Path to results directory
                - run_id: Run ID
                - pred_name: Prediction name
            parent_results_dir (Path): Parent directory for all results
            create_source_plots (bool): Whether to create per-source plots (default: False)
                                        Set to False to reduce runtime and only create comparison plots
        """
        super().__init__(
            "quantitative_eval", "Quantitative Evaluation", model_data, parent_results_dir
        )
        self.create_source_plots = create_source_plots

    def evaluate_sources(self, verbose=True, num_workers=0):
        """
        Args:
            verbose (bool): Whether to show progress bars.
            num_workers (int): Number of workers for parallel processing.
        """
        # Create a figure for comparing all models
        comparison_fig_dir = self.metric_results_dir / "comparisons"
        comparison_fig_dir.mkdir(exist_ok=True)

        # Process each model separately
        all_results = []

        for model_id, data in self.model_data.items():
            run_id = data["run_id"]
            pred_name = data["pred_name"]
            info_df = data["info_df"]

            print(f"\nEvaluating model: {run_id}, prediction: {pred_name}")

            # Create results directory for this model
            model_results_dir = self.model_dirs[model_id]["results_dir"]

            # Add n_available_sources column to info_df
            # Group by (batch_idx, index_in_batch) and count sources with avail=1
            available_sources = (
                info_df[info_df["avail"] == 1]
                .groupby(["batch_idx", "index_in_batch"])
                .size()
                .reset_index(name="n_available_sources")
            )

            # Merge the counts back to the original DataFrame
            info_df = pd.merge(
                info_df, available_sources, on=["batch_idx", "index_in_batch"], how="left"
            )

            # Fill NaN values with 0 (for rows where no sources are available)
            info_df["n_available_sources"] = info_df["n_available_sources"].fillna(0).astype(int)

            # Browse the batch indices in the DataFrame
            unique_batch_indices = info_df["batch_idx"].unique()

            # Create a DataFrame to store the results with columns:
            # source_name, index, dt, MSE, MAE, n_sources_available, channel_name
            results_df = pd.DataFrame(
                columns=[
                    "source_name",
                    "index",
                    "dt",
                    "MSE",
                    "MAE",
                    "n_sources_available",
                    "channel_name",
                    "run_id",
                    "pred_name",
                    "model_id",
                ]
            )
            results_csv = model_results_dir / "results.csv"
            results_df.to_csv(results_csv, index=False)

            if num_workers < 2:
                # Process batches sequentially
                for batch_idx in tqdm(unique_batch_indices, desc="Batches", disable=not verbose):
                    # Get the sources included in the batch
                    batch_info = info_df[info_df["batch_idx"] == batch_idx]
                    source_index_pairs = list(zip(batch_info["source_name"], batch_info["index"]))

                    # For each source-index pair, compute MSE and MAE metrics
                    for source_name, index in source_index_pairs:
                        source_info = batch_info[
                            (batch_info["source_name"] == source_name)
                            & (batch_info["index"] == index)
                        ]

                        if len(source_info) > 0:
                            targets, preds = self.load_batch(
                                source_name, index, batch_idx, model_id
                            )
                            if targets is not None and preds is not None:
                                compute_metrics(
                                    source_info,
                                    source_name,
                                    index,
                                    targets,
                                    preds,
                                    results_csv,
                                    run_id,
                                    pred_name,
                                    model_id,
                                )
            else:
                # Divide the dataframe into chunks for parallel processing
                batch_chunks = np.array_split(unique_batch_indices, num_workers)

                # Print the number of chunks
                if verbose:
                    print(f"Dividing work into {len(batch_chunks)} chunks")

                # Process each chunk in parallel
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for worker_id, chunk in enumerate(batch_chunks):
                        # Submit the chunk to be processed by a worker
                        future = executor.submit(
                            process_batch_chunk,
                            self,
                            info_df,
                            chunk,
                            results_csv,
                            verbose,
                            worker_id,
                            model_id,
                            run_id,
                            pred_name,
                        )
                        futures.append(future)

                    # Wait for all futures to complete
                    for future in tqdm(futures, desc="Processing chunks", disable=not verbose):
                        future.result()

            # Load the results and compute aggregated metrics
            results_df = pd.read_csv(results_csv)

            # Store results for comparison
            results_df["model_id"] = model_id
            all_results.append(results_df)

            # Compute and display results for this model
            self.compute_and_display_results(results_df, model_results_dir)

        # Combine all results for comparison
        if len(all_results) > 0:
            combined_results = pd.concat(all_results, ignore_index=True)
            self.create_model_comparison_plots(combined_results, comparison_fig_dir)
        else:
            print("No results to compare!")

    def compute_and_display_results(self, results_df, results_dir):
        """Computes aggregate statistics and displays histogram figures for each source.

        Args:
            results_df (pd.DataFrame): DataFrame with the evaluation results.
            results_dir (Path): Directory to save results for this model.
        """
        # Create figures directory if it doesn't exist
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Calculate RMSE from MSE (square root of the mean of MSE values)
        # Group by source_name and channel_name
        source_channel_groups = results_df.groupby(["source_name", "channel_name"])

        # Compute RMSE for each group (square root of mean MSE)
        rmse_by_source_channel = (
            source_channel_groups["MSE"].mean().apply(np.sqrt).reset_index(name="RMSE")
        )

        # Overall statistics for MAE and RMSE
        overall_mean_mae = results_df["MAE"].mean()
        overall_std_mae = results_df["MAE"].std()
        overall_mean_rmse = rmse_by_source_channel["RMSE"].mean()

        print("\nError Statistics:")
        print(f"  Mean RMSE: {overall_mean_rmse:.6f}")
        print(f"  Mean MAE: {overall_mean_mae:.6f}")
        print(f"  Std MAE:  {overall_std_mae:.6f}")
        print("\nPer-Source Error Statistics:")

        # Get channels and sources
        channels = results_df["channel_name"].unique()
        sources = results_df["source_name"].unique()
        n_channels = len(channels)
        n_sources = len(sources)

        # Set a color palette with enough distinct colors
        colors = sns.color_palette("colorblind", n_sources)

        # Print statistics for each source and channel for all cases
        for channel in channels:
            channel_data = rmse_by_source_channel[
                rmse_by_source_channel["channel_name"] == channel
            ]

            for _, row in channel_data.iterrows():
                source_name = row["source_name"]
                rmse_value = row["RMSE"]
                sample_count = len(
                    results_df[
                        (results_df["source_name"] == source_name)
                        & (results_df["channel_name"] == channel)
                    ]
                )

                print(f"  {source_name} - {channel}:")
                print(f"    RMSE: {rmse_value:.6f}")
                print(f"    Sample count: {sample_count}")

        # Only create source-wise plots if requested
        if self.create_source_plots:
            # For RMSE - Bar chart since it's a single value per source/channel
            fig_height = max(3, 1.5 * n_channels)  # Adaptive height based on number of channels
            fig_width = min(8, max(5, 2 + 0.5 * n_sources))  # Adaptive width

            plt.figure(figsize=(fig_width, fig_height))

            for i, channel in enumerate(channels):
                # Create subplot with shared x-axis if multiple channels
                if n_channels > 1:
                    ax = plt.subplot(n_channels, 1, i + 1)
                else:
                    ax = plt.subplot(1, 1, 1)

                # Filter data for this channel
                channel_data = rmse_by_source_channel[
                    rmse_by_source_channel["channel_name"] == channel
                ]

                # Create bar plot for RMSE values
                bars = ax.bar(
                    channel_data["source_name"],
                    channel_data["RMSE"],
                    color=colors[: len(channel_data)],
                )

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + height * 0.01,
                        f"{height:.2f}",  # Format to 2 decimal places
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=0,
                    )

                if n_channels > 1:
                    plt.title(f"{channel}", fontsize=10)
                else:
                    plt.title(f"RMSE by Source - {channel}", fontsize=11)

                plt.xlabel("" if i < n_channels - 1 else "Source")
                plt.ylabel("RMSE")
                plt.xticks(rotation=30, ha="right")

                # Use ScalarFormatter instead of FormatStrFormatter, which is compatible with ticklabel_format
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                # Disable scientific notation
                plt.ticklabel_format(style="plain", axis="y")

                plt.grid(True, alpha=0.3, linestyle="--", axis="y")

            plt.tight_layout()
            rmse_path = figures_dir / "rmse_by_source_channel.pdf"
            plt.savefig(rmse_path, bbox_inches="tight")
            # Also save as PNG for easy viewing
            plt.savefig(figures_dir / "rmse_by_source_channel.png", dpi=300, bbox_inches="tight")
            plt.close()

            # For MAE - Compact density plots (keep this part from the original)
            plt.figure(figsize=(7, fig_height))

            for i, channel in enumerate(channels):
                channel_data = results_df[results_df["channel_name"] == channel]

                if n_channels > 1:
                    ax = plt.subplot(n_channels, 1, i + 1)
                else:
                    ax = plt.subplot(1, 1, 1)

                for j, source_name in enumerate(sources):
                    source_data = channel_data[channel_data["source_name"] == source_name]
                    if len(source_data) > 0:
                        mean_mae = source_data["MAE"].mean()

                        # Plot compact KDE
                        sns.kdeplot(
                            source_data["MAE"],
                            label=f"{source_name} (μ={mean_mae:.2f})",  # Format to 2 decimal places
                            color=colors[j],
                            fill=True,
                            alpha=0.3,
                            linewidth=1.5,
                        )

                if n_channels > 1:
                    plt.title(f"{channel}", fontsize=10)
                else:
                    plt.title(f"MAE Distribution - {channel}", fontsize=11)

                plt.xlabel("MAE" if i == n_channels - 1 else "")
                plt.ylabel("Density")

                # Use ScalarFormatter instead of FormatStrFormatter, which is compatible with ticklabel_format
                ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                # Disable scientific notation
                plt.ticklabel_format(style="plain", axis="x")

                # Only show legend in the first subplot if multiple channels
                if i == 0 or n_channels == 1:
                    plt.legend(fontsize=8, frameon=True, framealpha=0.7, loc="best")

                plt.grid(True, alpha=0.3, linestyle="--")

            plt.tight_layout()
            mae_hist_path = figures_dir / "mae_density_by_source_channel.pdf"
            plt.savefig(mae_hist_path, bbox_inches="tight")
            plt.savefig(
                figures_dir / "mae_density_by_source_channel.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            print(f"\nDensity plots saved to: {figures_dir}")

            # MAE box plot - similar compact style
            plt.figure(figsize=(fig_width, fig_height))

            for i, channel in enumerate(channels):
                if n_channels > 1:
                    ax = plt.subplot(n_channels, 1, i + 1)
                else:
                    ax = plt.subplot(1, 1, 1)

                channel_data = results_df[results_df["channel_name"] == channel]

                # Create more compact, publication-ready boxplot with proper hue
                boxplot = sns.boxplot(
                    x="source_name",
                    y="MAE",
                    data=channel_data,
                    width=0.6,
                    hue="source_name",
                    palette=colors,
                    legend=False,  # No legend needed since x-axis already shows sources
                    showfliers=False,  # Hide outliers for cleaner appearance
                    medianprops={"color": "black"},
                    ax=ax,
                    order=sources,  # Use sources instead of model_ids for source-wise plots
                )

                # Add median value labels to the top-right of each box
                if len(channel_data) > 0:
                    # For each source, calculate and display the median
                    for j, source_name in enumerate(sources):
                        source_data = channel_data[channel_data["source_name"] == source_name]
                        if len(source_data) > 0:
                            median_val = source_data["MAE"].median()
                            # Get the position of the box
                            q1 = source_data["MAE"].quantile(0.25)
                            q3 = source_data["MAE"].quantile(0.75)
                            # Position the text at the top-right of the box
                            x_pos = j + 0.25  # Slightly to the right of the box center
                            y_pos = q3 + (q3 - q1) * 0.1  # Slightly above the top of the box
                            ax.text(
                                x_pos,  # X position (slightly right of box center)
                                y_pos,  # Y position (slightly above the top of the box)
                                f"{median_val:.2f}",  # Format to 2 decimal places
                                ha="left",
                                va="bottom",
                                fontsize=7,
                                color="black",
                                weight="bold",
                                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                            )

                if n_channels > 1:
                    plt.title(f"{channel}", fontsize=10)
                else:
                    plt.title(f"MAE Distribution - {channel}", fontsize=11)

                plt.xlabel("" if i < n_channels - 1 else "Source")
                plt.ylabel("MAE")
                plt.xticks(rotation=30, ha="right")

                # Use ScalarFormatter instead of FormatStrFormatter, which is compatible with ticklabel_format
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                # Disable scientific notation
                plt.ticklabel_format(style="plain", axis="y")

                plt.grid(True, alpha=0.3, linestyle="--", axis="y")

            plt.tight_layout()
            mae_box_path = figures_dir / "mae_boxplot_by_source_channel.pdf"
            plt.savefig(mae_box_path, bbox_inches="tight")
            plt.savefig(
                figures_dir / "mae_boxplot_by_source_channel.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            print(f"Publication-ready box plots saved to: {figures_dir}")
        else:
            print("\nSkipping source-wise plots as requested")

        # Save summary statistics to CSV
        # Include both MSE and RMSE in the summary
        mse_stats = (
            source_channel_groups[["MSE", "MAE"]]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

        # Add RMSE to summary statistics - no need to rename since it's already called "RMSE"
        summary_stats = pd.merge(
            mse_stats,
            rmse_by_source_channel[["source_name", "channel_name", "RMSE"]],
            on=["source_name", "channel_name"],
            how="left",
        )

        summary_stats_path = results_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_stats_path, index=False)

        print(f"Summary statistics saved to: {summary_stats_path}")

    def create_model_comparison_plots(self, combined_results, comparison_dir):
        """Creates plots comparing all models

        Args:
            combined_results (pd.DataFrame): Combined results from all models
            comparison_dir (Path): Directory to save comparison plots
        """
        if len(combined_results) == 0:
            return

        # Get model IDs in a consistent order (sorted) to ensure consistency across all plots
        model_ids = sorted(combined_results["model_id"].unique())
        n_models = len(model_ids)

        # Continue even if there's only one model
        print("\nCreating model comparison plots...")
        channels = combined_results["channel_name"].unique()
        n_channels = len(channels)

        # Setup figure sizes based on content
        fig_height = max(3, 1.5 * n_channels)  # Adaptive height
        fig_width = min(8, max(5, 2 + 0.5 * n_models))  # Adaptive width

        # Color palette - create a consistent mapping of model IDs to colors
        colors = sns.color_palette("colorblind", max(n_models, 2))  # Ensure at least 2 colors
        color_dict = dict(zip(model_ids, colors))

        # Calculate RMSE from MSE for each model and channel
        # Group by model_id, source_name, and channel_name to compute RMSE
        model_source_channel_groups = combined_results.groupby(
            ["model_id", "source_name", "channel_name"]
        )
        # Calculate RMSE (square root of mean MSE)
        rmse_values = (
            model_source_channel_groups["MSE"].mean().apply(np.sqrt).reset_index(name="RMSE")
        )

        # Now group by model_id and channel_name for the comparison plots
        model_channel_groups = combined_results.groupby(["model_id", "channel_name"])
        model_mean_rmse = (
            model_channel_groups["MSE"].mean().apply(np.sqrt).reset_index(name="RMSE_mean")
        )
        model_mean_mae = model_channel_groups["MAE"].mean().reset_index()

        # Get standard error for error bars (for MAE only since RMSE is a single value)
        model_std_mae = model_channel_groups["MAE"].sem().reset_index()

        # Merge mean and std for MAE plotting
        model_mae_stats = pd.merge(
            model_mean_mae,
            model_std_mae,
            on=["model_id", "channel_name"],
            suffixes=("_mean", "_sem"),
        )

        # 1. RMSE bar chart
        plt.figure(figsize=(fig_width, fig_height))

        for i, channel in enumerate(channels):
            if n_channels > 1:
                ax = plt.subplot(n_channels, 1, i + 1)
            else:
                ax = plt.subplot(1, 1, 1)

            channel_data = model_mean_rmse[model_mean_rmse["channel_name"] == channel]

            # Ensure data is sorted by our model_ids order
            channel_data = channel_data.set_index("model_id").loc[model_ids].reset_index()

            # Create bar plot for RMSE values
            bars = ax.bar(
                channel_data["model_id"],
                channel_data["RMSE_mean"],
                color=[color_dict[model_id] for model_id in channel_data["model_id"]],
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{height:.2f}",  # Format to 2 decimal places
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

            if n_channels > 1:
                plt.title(f"{channel}", fontsize=10)
            else:
                plt.title(f"RMSE by Model - {channel}", fontsize=11)

            plt.xlabel("" if i < n_channels - 1 else "Model")
            plt.ylabel("RMSE")
            # Only show x-axis tick labels for the bottom subplot
            if i < n_channels - 1:
                # First get the tick positions, then set empty labels
                ticks = plt.xticks()[0]
                plt.xticks(ticks, labels=[], rotation=30, ha="right")
            else:
                plt.xticks(rotation=30, ha="right")

            # Remove scientific notation for y-axis
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            # Disable scientific notation
            plt.ticklabel_format(style="plain", axis="y")

            plt.grid(True, alpha=0.3, linestyle="--", axis="y")

        plt.tight_layout()
        rmse_path = comparison_dir / "model_rmse_comparison.pdf"
        plt.savefig(rmse_path, bbox_inches="tight")
        plt.savefig(comparison_dir / "model_rmse_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. MAE boxplot
        plt.figure(figsize=(fig_width, fig_height))

        for i, channel in enumerate(channels):
            if n_channels > 1:
                ax = plt.subplot(n_channels, 1, i + 1)
            else:
                ax = plt.subplot(1, 1, 1)

            channel_data = combined_results[combined_results["channel_name"] == channel]

            # Create more compact, publication-ready boxplot with proper hue
            boxplot = sns.boxplot(
                x="model_id",
                y="MAE",
                data=channel_data,
                width=0.6,
                hue="model_id",
                palette=color_dict,
                showfliers=False,  # Hide outliers for cleaner appearance
                medianprops={"color": "black"},
                legend=False,
                ax=ax,
                order=model_ids,  # Explicitly set order to match our sorted model_ids
            )

            # Add median value labels to the top-right of each box
            if len(channel_data) > 0:
                # For each model, calculate and display the median
                for j, model_id in enumerate(model_ids):
                    model_data = channel_data[channel_data["model_id"] == model_id]
                    if len(model_data) > 0:
                        median_val = model_data["MAE"].median()
                        # Get the position of the box
                        q1 = model_data["MAE"].quantile(0.25)
                        q3 = model_data["MAE"].quantile(0.75)
                        # Position the text at the top-right of the box
                        x_pos = j + 0.25  # Slightly to the right of the box center
                        y_pos = q3 + (q3 - q1) * 0.1  # Slightly above the top of the box
                        ax.text(
                            x_pos,  # X position (slightly right of box center)
                            y_pos,  # Y position (slightly above the top of the box)
                            f"{median_val:.2f}",  # Format to 2 decimal places
                            ha="left",
                            va="bottom",
                            fontsize=7,
                            color="black",
                            weight="bold",
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                        )

            if n_channels > 1:
                plt.title(f"{channel}", fontsize=10)
            else:
                plt.title(f"MAE Distribution - {channel}", fontsize=11)

            plt.xlabel("" if i < n_channels - 1 else "Model")
            plt.ylabel("MAE")
            # Only show x-axis tick labels for the bottom subplot
            if i < n_channels - 1:
                # First get the tick positions, then set empty labels
                ticks = plt.xticks()[0]
                plt.xticks(ticks, labels=[], rotation=30, ha="right")
            else:
                plt.xticks(rotation=30, ha="right")

            # Use ScalarFormatter instead of FormatStrFormatter, which is compatible with ticklabel_format
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            # Disable scientific notation
            plt.ticklabel_format(style="plain", axis="y")

            plt.grid(True, alpha=0.3, linestyle="--", axis="y")

        plt.tight_layout()
        mae_box_path = comparison_dir / "model_mae_comparison_boxplot.pdf"
        plt.savefig(mae_box_path, bbox_inches="tight")
        plt.savefig(
            comparison_dir / "model_mae_comparison_boxplot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Publication-ready model comparison plots saved to: {comparison_dir}")


def compute_metrics(
    source_info, source_name, index, targets, preds, results_csv, run_id, pred_name, model_id
):
    """Computes MSE and MAE between targets and predictions for all common channels.

    Args:
        source_info (pd.DataFrame): DataFrame with at least the following columns:
            avail, batch_idx, index_in_batch, dt.
        source_name (str): Name of the source.
        index (int): Index of the source.
        targets (xarray.Dataset): Target dataset.
        preds (xarray.Dataset): Predictions dataset.
        results_csv (Path): Path to the CSV file where results will be saved.
        run_id (str): Run ID of the model
        pred_name (str): Prediction name
        model_id (str): Combined model identifier
    """
    # Filter out samples where avail is not 0
    mask = source_info["avail"] == 0
    if not mask.any():
        return  # No samples to process

    # Get indices of samples to process
    sample_indices = np.where(mask)[0]

    # Filter source_info to only include rows where avail is 0
    filtered_source_info = source_info.loc[mask].reset_index(drop=True)

    # Get the dt and n_available_sources for the samples
    dt_values = targets.isel(samples=sample_indices)["dt"].values
    n_available_sources = filtered_source_info["n_available_sources"].values

    # Select only the samples we need to process
    filtered_targets = targets.isel(samples=sample_indices)
    filtered_preds = preds.isel(samples=sample_indices)

    # Apply post-processing to targets and predictions based on source type
    filtered_targets, filtered_preds = apply_post_processing(
        filtered_targets, filtered_preds, source_name
    )

    # Find common channels/variables between targets and predictions
    common_vars = set(filtered_targets.data_vars).intersection(set(filtered_preds.data_vars))

    # Exclude 'dt', 'lat', 'lon' from the metrics calculation if they exist
    exclude_vars = {"dt", "lat", "lon", "longitude", "latitude"}
    channel_names = [var for var in common_vars if var not in exclude_vars]

    results_list = []

    # Compute MSE and MAE for each channel across all samples
    for channel in channel_names:
        # Extract target and prediction data for this channel
        target_data = filtered_targets[channel].values
        pred_data = filtered_preds[channel].values

        # For each sample, compute MSE and MAE
        for i in range(len(sample_indices)):
            # Extract the sample data
            sample_target = target_data[i]
            sample_pred = pred_data[i]

            # Compute MSE and MAE regardless of channel dimensions/shape
            # Flatten arrays to handle any dimensionality
            sample_target_flat = sample_target.flatten()
            sample_pred_flat = sample_pred.flatten()

            # Compute difference and absolute difference (NaN will propagate)
            diff_squared = (sample_target_flat - sample_pred_flat) ** 2
            abs_diff = np.abs(sample_target_flat - sample_pred_flat)

            # Use np.nanmean to calculate metrics, automatically ignoring NaN values
            mse = np.nanmean(diff_squared)
            mae = np.nanmean(abs_diff)

            # Skip if all values were NaN (resulting in NaN metrics)
            if np.isnan(mse) or np.isnan(mae):
                continue

            # Append results for this sample and channel
            results_list.append(
                {
                    "source_name": source_name,
                    "index": index,
                    "dt": dt_values[i],
                    "MSE": mse,
                    "MAE": mae,
                    "n_sources_available": n_available_sources[i],
                    "channel_name": channel,
                    "run_id": run_id,
                    "pred_name": pred_name,
                    "model_id": model_id,
                }
            )

    # Create a DataFrame with all results and append to CSV
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(results_csv, mode="a", header=False, index=False)


def process_batch_chunk(
    evaluator,
    info_df,
    batch_indices,
    results_csv,
    verbose,
    worker_id=0,
    model_id=None,
    run_id=None,
    pred_name=None,
):
    """Process a chunk of batch indices in a single worker process.

    Args:
        evaluator: The QuantitativeEvaluation instance
        info_df (pd.DataFrame): The complete info dataframe
        batch_indices (list): List of batch indices to process in this worker
        results_csv (Path): Path to the CSV file where results will be saved
        verbose (bool): Whether to show progress bars
        worker_id (int): ID of the worker processing this chunk
        model_id (str): Combined model identifier
        run_id (str): Run ID of the model
        pred_name (str): Prediction name
    Returns:
        bool: True indicating successful completion
    """
    # Only show progress bar for the first worker
    show_progress = verbose and worker_id == 0

    for batch_idx in tqdm(
        batch_indices, desc=f"Processing batch (chunk {worker_id})", disable=not show_progress
    ):
        # Get the sources included in the batch
        batch_info = info_df[info_df["batch_idx"] == batch_idx]
        source_index_pairs = list(zip(batch_info["source_name"], batch_info["index"]))

        # For each source-index pair, compute MSE and MAE metrics
        for source_name, index in source_index_pairs:
            source_info = batch_info[
                (batch_info["source_name"] == source_name) & (batch_info["index"] == index)
            ]

            if len(source_info) > 0:
                targets, preds = evaluator.load_batch(source_name, index, batch_idx, model_id)
                if targets is not None and preds is not None:
                    compute_metrics(
                        source_info,
                        source_name,
                        index,
                        targets,
                        preds,
                        results_csv,
                        run_id,
                        pred_name,
                        model_id,
                    )

    return True  # Indicate successful completion


# POST-PROCESSING FUNCTIONS
# These are function that apply a specific post-processing to predictions
# and targets depending on the source, before computing the metrics.
def apply_post_processing(targets, preds, source_name):
    """Apply post-processing to targets and predictions based on source type.

    Args:
        targets (xarray.Dataset): Target dataset
        preds (xarray.Dataset): Predictions dataset
        source_name (str): Name of the source

    Returns:
        tuple: Processed (targets, preds)
    """
    if source_name == "tc_primed_era5":
        return process_era5(targets, preds)
    elif source_name == "tc_primed_storm_metadata":
        return process_storm_metadata(targets, preds)
    elif "pmw" in source_name:
        return process_pmw(targets, preds)
    else:
        # For other sources, return the datasets unchanged
        return targets, preds


def process_storm_metadata(targets, preds):
    """Post-process storm metadata by renaming the 'intensity' variable to 'Maximum Sustained Windspeed (kt)'.

    Args:
        targets (xarray.Dataset): Target dataset
        preds (xarray.Dataset): Predictions dataset

    Returns:
        tuple: Processed (targets, preds)
    """
    # Rename the 'intensity' variable if it exists
    if "intensity" in targets.data_vars:
        targets = targets.rename({"intensity": "Maximum Sustained Windspeed (kt)"})

    if "intensity" in preds.data_vars:
        preds = preds.rename({"intensity": "Maximum Sustained Windspeed (kt)"})

    return targets, preds


def process_era5(targets, preds, radius_around_center_km=1000):
    """Post-process ERA5 data by computing the 10m wind speed from the u and v components
    but keeping the original components for evaluation.
    """
    # First compute the derived values using the original variable names
    targets["mslp"] = targets["pressure_msl"].min(dim=["H", "W"])
    preds["mslp"] = preds["pressure_msl"].min(dim=["H", "W"])

    # Compute the wind speed from the u and v components
    targets["wind_speed_10m"] = np.sqrt(targets["u_wind_10m"] ** 2 + targets["v_wind_10m"] ** 2)
    preds["wind_speed_10m"] = np.sqrt(preds["u_wind_10m"] ** 2 + preds["v_wind_10m"] ** 2)

    # Now rename all variables at once
    rename_dict = {
        "pressure_msl": "Mean sea-level pressure",
        "mslp": "Minimum sea-level pressure",
        "dist_to_center": "Distance to center",  # We'll still rename it, but drop it later
        "u_wind_10m": "U-wind 10m",
        "v_wind_10m": "V-wind 10m",
        "wind_speed_10m": "Wind speed 10m",
    }

    targets = targets.rename(rename_dict)
    preds = preds.rename(rename_dict)

    # Drop the "Distance to center" variable after it's been used
    if "Distance to center" in targets.data_vars:
        targets = targets.drop_vars("Distance to center")
    if "Distance to center" in preds.data_vars:
        preds = preds.drop_vars("Distance to center")

    return targets, preds


def process_pmw(targets, preds, radius_around_center_km=1000):
    """Post-process PMW data by:
    - Renaming the variable whose name contains "TB" to "Brightness Temperature".
    - Setting every point in the "Brightness Temperature" channel to NaN
        outside a radius around the center.
    - Removing the distance to center variable after it's been used for filtering.
    """
    # Rename the variable whose name contains "TB" to "Brightness_Temperature"
    tb_var = [var for var in targets.data_vars if "TB" in var]
    if len(tb_var) == 1:
        targets = targets.rename({tb_var[0]: "Brightness Temperature"})
        preds = preds.rename({tb_var[0]: "Brightness Temperature"})
    else:
        raise ValueError("More than one variable containing 'TB' found.")

    if "dist_to_center" in targets.data_vars:
        # Compute the distance mask: every point for which the "dist_to_center"
        # channel is greater than the radius is set to NaN
        dist_to_center = targets["dist_to_center"]
        dist_mask = dist_to_center > radius_around_center_km

        # Set the values outside the radius to NaN
        targets["Brightness Temperature"] = targets["Brightness Temperature"].where(~dist_mask)
        preds["Brightness Temperature"] = preds["Brightness Temperature"].where(~dist_mask)

        # After using dist_to_center for filtering, drop it so it's not included in error calculations
        targets = targets.drop_vars("dist_to_center")
        preds = preds.drop_vars("dist_to_center")

    # Also check if the renamed version exists (in case it was already renamed)
    if "Distance to center" in targets.data_vars:
        targets = targets.drop_vars("Distance to center")
    if "Distance to center" in preds.data_vars:
        preds = preds.drop_vars("Distance to center")

    return targets, preds
