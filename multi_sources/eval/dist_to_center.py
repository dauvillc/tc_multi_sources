"""
Implements the DistanceToCenter class.
"""

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from haversine import Unit, haversine_vector
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


class DistanceToCenter(AbstractMultisourceEvaluationMetric):
    """Finds the minima in the dist_to_center field of the predictions and targets,
    and computes the distance (in km) between them.
    """

    def __init__(self, model_data, parent_results_dir, create_source_plots=False):
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
            "dist_to_center_eval", "Distance to Center Evaluation", model_data, parent_results_dir
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
            # source_name, index, dt, error_km, n_sources_available, lat_target, lon_target, lat_pred, lon_pred
            results_df = pd.DataFrame(
                columns=[
                    "source_name",
                    "index",
                    "dt",
                    "error_km",
                    "n_sources_available",
                    "lat_target",
                    "lon_target",
                    "lat_pred",
                    "lon_pred",
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

                    # For each source-index pair, process the dist_to_center
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
                                process_dist_to_center(
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

        # Overall statistics
        overall_mean = results_df["error_km"].mean()
        overall_std = results_df["error_km"].std()
        overall_median = results_df["error_km"].median()

        print("\nError Statistics:")
        print(f"  Mean error: {overall_mean:.2f} km")
        print(f"  Median error: {overall_median:.2f} km")
        print(f"  Std error:  {overall_std:.2f} km")
        print("\nPer-Source Error Statistics:")

        # Group by source name for per-source statistics
        source_groups = results_df.groupby("source_name")

        # Get list of sources
        sources = results_df["source_name"].unique()
        n_sources = len(sources)

        # Set a color palette with enough distinct colors
        colors = sns.color_palette("colorblind", n_sources)

        # Print statistics for each source
        for source_name, group in source_groups:
            mean_error = group["error_km"].mean()
            std_error = group["error_km"].std()
            median_error = group["error_km"].median()

            # Print statistics for this source
            print(f"  {source_name}:")
            print(f"    Mean error: {mean_error:.2f} km")
            print(f"    Median error: {median_error:.2f} km")
            print(f"    Std error:  {std_error:.2f} km")
            print(f"    Sample count: {len(group)}")

        # Only create source-wise plots if requested
        if self.create_source_plots:
            # Setup the figure for KDE plots (more elegant than histograms for publications)
            plt.figure(figsize=(7, 5))

            # Plot KDE for each source
            for i, (source_name, group) in enumerate(source_groups):
                mean_error = group["error_km"].mean()
                median_error = group["error_km"].median()

                # Plot KDE with fill for professional appearance
                sns.kdeplot(
                    group["error_km"],
                    label=f"{source_name} (median={median_error:.1f}km)",
                    color=colors[i],
                    fill=True,
                    alpha=0.3,
                    linewidth=1.5,
                )

            plt.title("Distribution of Distance Errors by Source", fontsize=11)
            plt.xlabel("Error (km)")
            plt.ylabel("Density")
            plt.legend(frameon=True, framealpha=0.7, fontsize=8)
            plt.grid(True, alpha=0.3, linestyle="--")

            # Add a vertical line at x=0 as reference
            plt.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)

            # Save the figure in both PDF (for publication) and PNG formats
            plt.tight_layout()
            plt.savefig(figures_dir / "error_density_by_source.pdf", bbox_inches="tight")
            plt.savefig(figures_dir / "error_density_by_source.png", dpi=300, bbox_inches="tight")
            plt.close()

            print(f"\nDensity plot saved to: {figures_dir}")

            # Create a publication-ready box plot to compare error distributions
            fig_width = min(
                8, max(5, 2 + 0.5 * n_sources)
            )  # Adaptive width based on number of sources
            plt.figure(figsize=(fig_width, 5))

            # Create more informative boxplot
            boxplot = sns.boxplot(
                x="source_name",
                y="error_km",
                data=results_df,
                width=0.6,
                hue="source_name",
                palette=colors,
                showfliers=False,  # Hide outliers for cleaner appearance
                medianprops={"color": "black"},
                legend=False,
            )

            # Add individual points as a swarm plot with small size
            sns.stripplot(
                x="source_name",
                y="error_km",
                data=results_df,
                size=2.5,
                alpha=0.4,
                jitter=True,
                hue="source_name",
                palette=colors,
                legend=False,
            )

            # Add median value labels to the top of each box
            for j, source_name in enumerate(sources):
                source_data = results_df[results_df["source_name"] == source_name]
                if len(source_data) > 0:
                    median_val = source_data["error_km"].median()
                    # Get the position of the box
                    q1 = source_data["error_km"].quantile(0.25)
                    q3 = source_data["error_km"].quantile(0.75)
                    # Position the text above the box
                    y_pos = q3 + (q3 - q1) * 0.15  # Slightly above the top of the box
                    plt.text(
                        j,  # X position (center of box)
                        y_pos,  # Y position (above the box)
                        f"{median_val:.1f}",  # Format to 1 decimal place
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="black",
                        weight="bold",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
                    )

            plt.title("Distance Error Distribution by Source", fontsize=11)
            plt.xlabel("Source")
            plt.ylabel("Error (km)")
            plt.xticks(rotation=30, ha="right")
            plt.grid(True, alpha=0.3, linestyle="--", axis="y")

            # Save the box plot in both formats
            plt.tight_layout()
            plt.savefig(figures_dir / "error_boxplot_by_source.pdf", bbox_inches="tight")
            plt.savefig(figures_dir / "error_boxplot_by_source.png", dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Publication-ready box plot saved to: {figures_dir}")

            # Create scatter plot of predicted vs true position
            plt.figure(figsize=(7, 6))

            for i, (source_name, group) in enumerate(source_groups):
                # Plot arrows from true to predicted position
                for _, row in group.iterrows():
                    plt.arrow(
                        row["lon_target"],
                        row["lat_target"],
                        row["lon_pred"] - row["lon_target"],
                        row["lat_pred"] - row["lat_target"],
                        color=colors[i],
                        alpha=0.5,
                        length_includes_head=True,
                        head_width=0.1,
                        head_length=0.1,
                    )

                # Plot the true positions
                plt.scatter(
                    group["lon_target"],
                    group["lat_target"],
                    label=f"{source_name} (true)",
                    color=colors[i],
                    marker="o",
                    s=30,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Plot the predicted positions with a different marker
                plt.scatter(
                    group["lon_pred"],
                    group["lat_pred"],
                    label=f"{source_name} (pred)",
                    color=colors[i],
                    marker="x",
                    s=30,
                )

            plt.title("True vs Predicted Storm Center Positions", fontsize=11)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.grid(True, alpha=0.3, linestyle="--")

            # Add a legend with only one entry per source (combining true and pred)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(
                by_label.values(), by_label.keys(), frameon=True, framealpha=0.7, fontsize=8
            )

            # Save the position plot
            plt.tight_layout()
            plt.savefig(figures_dir / "position_scatter.pdf", bbox_inches="tight")
            plt.savefig(figures_dir / "position_scatter.png", dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Position scatter plot saved to: {figures_dir}")

            # Create a plot of error vs distance from center
            if "dt" in results_df.columns:
                plt.figure(figsize=(7, 5))

                for i, (source_name, group) in enumerate(source_groups):
                    plt.scatter(
                        group["dt"],
                        group["error_km"],
                        label=source_name,
                        color=colors[i],
                        alpha=0.7,
                        s=25,
                    )

                    # Add trend line
                    if len(group) > 1:
                        z = np.polyfit(group["dt"], group["error_km"], 1)
                        p = np.poly1d(z)
                        plt.plot(
                            sorted(group["dt"]),
                            p(sorted(group["dt"])),
                            color=colors[i],
                            linestyle="--",
                            alpha=0.8,
                        )

                plt.title("Error vs Time to Event", fontsize=11)
                plt.xlabel("Time to Event (hours)")
                plt.ylabel("Error (km)")
                plt.grid(True, alpha=0.3, linestyle="--")
                plt.legend(frameon=True, framealpha=0.7, fontsize=8)

                # Save the time plot
                plt.tight_layout()
                plt.savefig(figures_dir / "error_vs_time.pdf", bbox_inches="tight")
                plt.savefig(figures_dir / "error_vs_time.png", dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Error vs time plot saved to: {figures_dir}")
        else:
            print("\nSkipping source-wise plots as requested")

        # Save summary statistics to CSV
        summary_stats = (
            source_groups["error_km"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
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
        if len(model_ids) <= 1:
            return

        print("\nCreating model comparison plots...")

        # Setup parameters for publication-quality figures
        n_models = len(model_ids)
        fig_width = min(8, max(5, 2 + 0.5 * n_models))  # Adaptive width

        # Create a consistent color palette and mapping
        colors = sns.color_palette("colorblind", n_models)
        color_dict = dict(zip(model_ids, colors))

        # 1. Overall error comparison by model - publication-ready boxplot
        plt.figure(figsize=(fig_width, 5))

        # Create a more informative boxplot
        boxplot = sns.boxplot(
            x="model_id",
            y="error_km",
            data=combined_results,
            width=0.6,
            hue="model_id",
            palette=color_dict,
            showfliers=False,  # Hide outliers for cleaner appearance
            medianprops={"color": "black"},
            legend=False,
            order=model_ids,  # Explicitly set order to match our sorted model_ids
        )

        # Add individual points for better visualization
        if len(combined_results) < 200:  # Only add points if not too many
            sns.stripplot(
                x="model_id",
                y="error_km",
                data=combined_results,
                size=2.5,
                alpha=0.4,
                jitter=True,
                hue="model_id",
                palette=color_dict,
                legend=False,
                order=model_ids,  # Explicitly set order to match our sorted model_ids
            )

        # Add median value labels to the top of each box
        for i, model_id in enumerate(model_ids):
            model_data = combined_results[combined_results["model_id"] == model_id]
            if len(model_data) > 0:
                median_val = model_data["error_km"].median()
                # Get the position of the box
                q1 = model_data["error_km"].quantile(0.25)
                q3 = model_data["error_km"].quantile(0.75)
                # Position the text above the box
                y_pos = q3 + (q3 - q1) * 0.15  # Slightly above the top of the box
                plt.text(
                    i,  # X position (center of box)
                    y_pos,  # Y position (above the box)
                    f"{median_val:.1f}",  # Format to 1 decimal place for cleaner appearance
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    weight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
                )

        plt.title("Error in Storm Center Location by Model", fontsize=11)
        plt.xlabel("Model")
        plt.ylabel("Error (km)")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, alpha=0.3, linestyle="--", axis="y")

        plt.tight_layout()
        plt.savefig(comparison_dir / "model_error_comparison_boxplot.pdf", bbox_inches="tight")
        plt.savefig(
            comparison_dir / "model_error_comparison_boxplot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Bar chart of mean errors by model with error bars
        model_groups = combined_results.groupby("model_id")
        model_mean = model_groups["error_km"].mean().reset_index()
        model_sem = model_groups["error_km"].sem().reset_index()

        # Merge mean and std for plotting
        model_stats = pd.merge(model_mean, model_sem, on="model_id", suffixes=("_mean", "_sem"))

        # Ensure data is sorted by our model_ids order
        model_stats = model_stats.set_index("model_id").loc[model_ids].reset_index()

        plt.figure(figsize=(fig_width, 5))

        # Create bar plot with error bars
        barplot = sns.barplot(
            x="model_id",
            y="error_km_mean",
            data=model_stats,
            hue="model_id",
            palette=color_dict,
            legend=False,
            order=model_ids,  # Explicitly set order to match our sorted model_ids
        )

        # Add error bars manually
        for i, bar in enumerate(barplot.patches):
            if i < len(model_stats):
                barplot.errorbar(
                    bar.get_x() + bar.get_width() / 2,
                    model_stats.iloc[i]["error_km_mean"],
                    yerr=model_stats.iloc[i]["error_km_sem"],
                    color="black",
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                )

        plt.title("Mean Error by Model", fontsize=11)
        plt.xlabel("Model")
        plt.ylabel("Mean Error (km)")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, alpha=0.3, linestyle="--", axis="y")

        plt.tight_layout()
        plt.savefig(comparison_dir / "model_mean_error_comparison.pdf", bbox_inches="tight")
        plt.savefig(
            comparison_dir / "model_mean_error_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Comparison table with additional statistics
        comparison_stats = (
            model_groups["error_km"]
            .agg(["mean", "median", "std", "sem", "min", "max", "count"])
            .reset_index()
        )
        comparison_stats.to_csv(comparison_dir / "model_comparison_statistics.csv", index=False)

        # 4. Create a density plot comparing error distributions across models
        plt.figure(figsize=(7, 5))

        for i, model_id in enumerate(model_ids):  # Use sorted model_ids for consistent order
            model_data = combined_results[combined_results["model_id"] == model_id]
            mean_error = model_data["error_km"].mean()
            median_error = model_data["error_km"].median()

            # Plot KDE with fill using the color dictionary
            sns.kdeplot(
                model_data["error_km"],
                label=f"{model_id} (median={median_error:.1f}km)",
                color=color_dict[model_id],  # Use the color dictionary
                fill=True,
                alpha=0.3,
                linewidth=1.5,
            )

        plt.title("Error Distribution by Model", fontsize=11)
        plt.xlabel("Error (km)")
        plt.ylabel("Density")
        plt.legend(frameon=True, framealpha=0.7, fontsize=8)
        plt.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(comparison_dir / "model_error_density.pdf", bbox_inches="tight")
        plt.savefig(comparison_dir / "model_error_density.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 5. Per-source comparison across models - publication-ready
        sources = combined_results["source_name"].unique()

        for source in sources:
            source_data = combined_results[combined_results["source_name"] == source]

            # Boxplot per source
            plt.figure(figsize=(fig_width, 5))

            boxplot = sns.boxplot(
                x="model_id",
                y="error_km",
                data=source_data,
                width=0.6,
                hue="model_id",
                palette=color_dict,
                showfliers=False,
                medianprops={"color": "black"},
                legend=False,
                order=model_ids,  # Explicitly set order to match our sorted model_ids
            )

            # Add individual points if not too many
            if len(source_data) < 200:
                sns.stripplot(
                    x="model_id",
                    y="error_km",
                    data=source_data,
                    size=2.5,
                    alpha=0.4,
                    jitter=True,
                    hue="model_id",
                    palette=color_dict,
                    legend=False,
                    order=model_ids,  # Explicitly set order to match our sorted model_ids
                )

            plt.title(f"Error Distribution: {source}", fontsize=11)
            plt.xlabel("Model")
            plt.ylabel("Error (km)")
            plt.xticks(rotation=30, ha="right")
            plt.grid(True, alpha=0.3, linestyle="--", axis="y")

            plt.tight_layout()
            plt.savefig(
                comparison_dir / f"source_{source}_model_comparison.pdf", bbox_inches="tight"
            )
            plt.savefig(
                comparison_dir / f"source_{source}_model_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Create a scatter plot of position predictions for this source
            plt.figure(figsize=(7, 6))

            for i, model_id in enumerate(model_ids):  # Use sorted model_ids for consistent order
                model_source_data = source_data[source_data["model_id"] == model_id]

                # Plot the true positions (only once, using the first model's data)
                if i == 0:
                    plt.scatter(
                        model_source_data["lon_target"],
                        model_source_data["lat_target"],
                        label="True",
                        color="black",
                        marker="o",
                        s=30,
                        alpha=0.7,
                    )

                # Plot the predicted positions with a different marker for each model
                plt.scatter(
                    model_source_data["lon_pred"],
                    model_source_data["lat_pred"],
                    label=f"{model_id}",
                    color=color_dict[model_id],  # Use the color dictionary
                    marker="x",
                    s=30,
                    alpha=0.7,
                )

            plt.title(f"Position Predictions: {source}", fontsize=11)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.grid(True, alpha=0.3, linestyle="--")
            plt.legend(frameon=True, framealpha=0.7, fontsize=8)

            plt.tight_layout()
            plt.savefig(
                comparison_dir / f"source_{source}_position_comparison.pdf", bbox_inches="tight"
            )
            plt.savefig(
                comparison_dir / f"source_{source}_position_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        print(f"Publication-ready model comparison plots saved to: {comparison_dir}")


def process_dist_to_center(
    source_info, source_name, index, targets, preds, results_csv, run_id, pred_name, model_id
):
    """Processes the dist_to_center data for a source-index pair.

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

    # Find the minimum point in the dist_to_center field for all samples at once
    target_min_idx = filtered_targets["dist_to_center"].argmin(dim=["H", "W"])
    pred_min_idx = filtered_preds["dist_to_center"].argmin(dim=["H", "W"])

    # Extract H and W coordinates for minimum points
    target_min_h = target_min_idx["H"].values
    target_min_w = target_min_idx["W"].values
    pred_min_h = pred_min_idx["H"].values
    pred_min_w = pred_min_idx["W"].values

    # Get lat/lon arrays
    target_lat = filtered_targets["lat"].values
    target_lon = filtered_targets["lon"].values
    pred_lat = filtered_preds["lat"].values
    pred_lon = filtered_preds["lon"].values

    # Create sample indices array
    n_samples = len(sample_indices)
    sample_idx = np.arange(n_samples)

    # Extract lat/lon at minimum points for all samples at once
    target_min_lat = target_lat[sample_idx, target_min_h, target_min_w]
    target_min_lon = target_lon[sample_idx, target_min_h, target_min_w]
    pred_min_lat = pred_lat[sample_idx, pred_min_h, pred_min_w]
    pred_min_lon = pred_lon[sample_idx, pred_min_h, pred_min_w]

    # Create arrays of coordinates for haversine_vector
    target_coords = np.column_stack((target_min_lat, target_min_lon))
    pred_coords = np.column_stack((pred_min_lat, pred_min_lon))

    # Use haversine_vector to compute all distances at once
    distances_km = haversine_vector(target_coords, pred_coords, Unit.KILOMETERS)

    # Create a DataFrame with all results
    results = pd.DataFrame(
        {
            "source_name": source_name,
            "index": index,
            "dt": dt_values,
            "error_km": distances_km,
            "n_sources_available": n_available_sources,
            "lat_target": target_min_lat,
            "lon_target": target_min_lon,
            "lat_pred": pred_min_lat,
            "lon_pred": pred_min_lon,
            "run_id": run_id,
            "pred_name": pred_name,
            "model_id": model_id,
        }
    )

    # Append all results to the CSV file at once
    results.to_csv(results_csv, mode="a", header=False, index=False)


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
        evaluator: The DistanceToCenter instance
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

        # For each source-index pair, process the dist_to_center data
        for source_name, index in source_index_pairs:
            source_info = batch_info[
                (batch_info["source_name"] == source_name) & (batch_info["index"] == index)
            ]

            if len(source_info) > 0:
                targets, preds = evaluator.load_batch(source_name, index, batch_idx, model_id)
                if targets is not None and preds is not None:
                    process_dist_to_center(
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
