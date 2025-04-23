"""
Implements the QuantitativeEvaluation class, which computes the mean squared error (MSE)
between the targets and predictions for a given source, for each channel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from string import Template
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.metrics import r2_score
from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


# A little code snippet from Shawn Chin & Peter Mortensen
# https://stackoverflow.com/questions/8906926/formatting-timedelta-objects/8907269#8907269
class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    """equivalent of strftime for timedelta objects"""
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


class QuantitativeEvaluation(AbstractMultisourceEvaluationMetric):
    """For each source, for each channel, computes the mean squared error (MSE) between
    the targets and predictions.
    """

    def __init__(self, predictions_dir, results_dir, displayed_names={}, displayed_units={}):
        """
        Args:
            predictions_dir (Path): Directory with the predictions.
            results_dir (Path): Directory where the results will be saved.
            displayed_names (dict of str to str): Dictionary {source_name: displayed_name}.
            displayed_units (dict of str to str): Dictionary {source_name: displayed_unit}.
        """
        super().__init__(
            "quantitative_eval", "Quantitative evaluation", predictions_dir, results_dir
        )
        self.displayed_names = displayed_names
        self.displayed_units = displayed_units

    def evaluate_sources(self, info_df, verbose=True, num_workers=0):
        """
        Args:
            info_df (pd.DataFrame): DataFrame with at least the following columns:
                source_name, avail, batch_idx, index_in_batch, dt.
            **kwargs: Additional keyword arguments.
        """
        # Browse the batch indices in the DataFrame
        unique_batch_indices = info_df["batch_idx"].unique()
        # Create a DataFrame to store the preds and targets with:
        # masked_source, pred, target, channel
        # and <source>_time_delta for every source.
        columns = ["masked_source", "channel", "pred", "target"]
        unique_sources = info_df["source_name"].unique()
        for source in unique_sources:
            columns.append(f"{source}_time_delta")
        results_df = pd.DataFrame(columns=columns)
        results_csv = self.results_dir / "results.csv"
        results_df.to_csv(results_csv, index=False)

        if num_workers < 2:
            for batch_idx in tqdm(unique_batch_indices, desc="Batches", disable=not verbose):
                # Get the sources included in the batch
                batch_info = info_df[info_df["batch_idx"] == batch_idx]
                sources = batch_info["source_name"].unique()
                # For each source, load the targets and predictions
                targets, preds = {}, {}
                for source in sources:
                    targets[source], preds[source] = self.load_batch(source, batch_idx)
                gather_results(batch_info, batch_idx, targets, preds, unique_sources, results_csv)
        else:
            # Divide the dataframe into chunks based on the number of workers
            batch_chunks = np.array_split(unique_batch_indices, num_workers)

            # Print the number of chunks
            if verbose:
                print(f"Dividing work into {len(batch_chunks)} chunks")

            # Process each chunk in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for worker_id, chunk in enumerate(batch_chunks):
                    # Submit the chunk of batch indices to be processed by a single worker
                    future = executor.submit(
                        process_batch_chunk,
                        self,
                        info_df,
                        chunk,
                        unique_sources,
                        results_csv,
                        verbose,
                        worker_id,
                    )
                    futures.append(future)

                # Wait for all futures to complete
                for future in tqdm(futures, desc="Processing chunks", disable=not verbose):
                    future.result()

        # Load the results csv and display the main aggregated metrics: RMSE, R2
        # MAE for each source and each channel
        results_df = pd.read_csv(results_csv)
        for col in results_df.columns:
            if col.endswith("_time_delta"):
                results_df[col] = pd.to_timedelta(results_df[col])
        grouped = results_df.groupby(["masked_source", "channel"])
        rmse = np.sqrt(grouped.apply(lambda x: np.mean((x["target"] - x["pred"]) ** 2)))
        r2 = grouped.apply(lambda x: r2_score(x["target"], x["pred"]))
        mae = grouped.apply(lambda x: np.mean(np.abs(x["target"] - x["pred"])))
        print("Results:")
        sources = results_df["masked_source"].unique()
        for source in sources:
            channels = results_df[results_df["masked_source"] == source]["channel"].unique()
            for channel in channels:
                print(f"Source: {source}, Channel: {channel}")
                print(f"RMSE: {rmse[source, channel]:.2f}")
                print(f"R2: {r2[source, channel]:.2f}")
                print(f"MAE: {mae[source, channel]:.2f}")
                print()

        aggregate_results(results_df, self.results_dir)

        # Compute and display the detailed results
        results_per_n_sources(
            unique_sources,
            results_df,
            self.results_dir,
            self.displayed_names,
            self.displayed_units,
        )


def gather_results(batch_info, batch_idx, targets, preds, all_sources, results_csv):
    """Gathers the results from the netcdf files to a single CSV file.
    Args:
        batch_info (pd.DataFrame): DataFrame with the information of the batch.
        batch_idx (int): Index of the batch to display.
        targets (dict): Dictionary with the targets for each source.
        preds (dict): Dictionary with the predictions for each source.
        all_sources (list): List with the unique sources in the dataset.
        results_csv (Path): Path to the CSV file where the results will be saved.
    """
    sources = batch_info["source_name"].unique()
    S = len(sources)
    # Unique indices in the batch
    batch_indices = batch_info["index_in_batch"].unique()
    for idx in batch_indices:
        for i, source in enumerate(sources):
            target_ds = targets[source]
            pred_ds = preds[source]
            if target_ds is not None:
                # Rows for each source in the sample
                sample_info = batch_info[batch_info["index_in_batch"] == idx]
                # Row of the sample corresponding to the source being looked at.
                sample_source_info = batch_info[batch_info["source_name"] == source].iloc[0]
                # The model only made a prediction when the sample was masked,
                # which corresponds to avail=0 (-1 for unavailable and 1 for available
                # but not masked).
                if pred_ds is not None and sample_source_info["avail"].item() == 0:
                    pred_sample = pred_ds.isel(samples=idx)
                    target_sample = target_ds.isel(samples=idx)
                    for channel in target_sample.data_vars:
                        target = target_sample[channel].values
                        pred = pred_sample[channel].values
                        # Save the results in a single-row DataFrame and append
                        # it to the results CSV
                        row = {
                            "masked_source": source,
                            "channel": channel,
                            "pred": pred,
                            "target": target,
                        }
                        for source in all_sources:
                            # Get the time delta for the source
                            source_info = sample_info[sample_info["source_name"] == source]
                            dt = pd.NaT
                            if len(source_info) > 0:
                                dt = source_info["dt"].values[0]
                                # If the source is not available, set the time delta to NaT
                                if source_info["avail"].values[0] == -1:
                                    dt = pd.NaT
                            row[f"{source}_time_delta"] = dt
                        results_df = pd.DataFrame([row])
                        results_df.to_csv(results_csv, mode="a", header=False, index=False)


def process_batch_chunk(
    evaluator, info_df, batch_indices, unique_sources, results_csv, verbose, worker_id=0
):
    """Process a chunk of batch indices in a single worker process.

    Args:
        evaluator: The QuantitativeEvaluation instance
        info_df (pd.DataFrame): The complete info dataframe
        batch_indices (list): List of batch indices to process in this worker
        unique_sources (list): List of all unique sources in the dataset
        results_csv (Path): Path to the CSV file where results will be saved
        verbose (bool): Whether to show progress bars
        worker_id (int): ID of the worker processing this chunk
    """
    # Only show progress bar for the first worker (worker_id=0)
    show_progress = verbose and worker_id == 0

    for batch_idx in tqdm(
        batch_indices, desc=f"Processing batch (chunk {worker_id})", disable=not show_progress
    ):
        # Get the sources included in the batch
        batch_info = info_df[info_df["batch_idx"] == batch_idx]
        sources = batch_info["source_name"].unique()

        # For each source, load the targets and predictions
        targets, preds = {}, {}
        for source in sources:
            targets[source], preds[source] = evaluator.load_batch(source, batch_idx)

        gather_results(batch_info, batch_idx, targets, preds, unique_sources, results_csv)

    return True  # Indicate successful completion


def aggregate_results(results_df, results_dir):
    """Aggregates the results from the CSV file:
    - RMSE, R2, MAE for each source and channel.
    Writes the results to a text file in the results directory.
    Args:
        results_df (pd.DataFrame): DataFrame with the results, with one row per sample.
        results_dir (Path): Directory where the results will be saved.
    """
    # Open a stream to a text file to write the results
    results_txt = results_dir / "results.txt"
    with open(results_txt, "w") as f:
        # For each source, channel and number of available sources, compute the RMSE, R2 and MAE
        # Store the results in a DataFrame with columns:
        # masked_source, channel, num_avail_sources, rmse, r2, mae
        grouped = results_df.groupby(["masked_source", "channel"])
        rmse = np.sqrt(grouped.apply(lambda x: np.mean((x["target"] - x["pred"]) ** 2)))
        r2 = grouped.apply(lambda x: r2_score(x["target"], x["pred"]))
        mae = grouped.apply(lambda x: np.mean(np.abs(x["target"] - x["pred"])))
        for source in results_df["masked_source"].unique():
            channels = results_df[results_df["masked_source"] == source]["channel"].unique()
            for channel in channels:
                f.write(f"Source: {source}, Channel: {channel}\n")
                f.write(f"RMSE: {rmse[source, channel]:.2f}\n")
                f.write(f"R2: {r2[source, channel]:.2f}\n")
                f.write(f"MAE: {mae[source, channel]:.2f}\n")
                f.write("\n")
        
    # Read the content of the text file and print it to the console
    with open(results_txt, "r") as f:
        content = f.read()
        print(content)


def results_per_n_sources(all_sources, results_df, figures_dir, displayed_names, displayed_units):
    """Displays the detailed results from the CSV file:
    - RMSE, R2, MAE for each source and channel versus the number of available sources
        in the sample.
    Args:
        all_sources (list): List with the unique sources in the dataset.
        results_df (pd.DataFrame): DataFrame with the results, with one row per sample.
        figures_dir (Path): Directory where the figures will be saved.
        displayed_names (dict of str to str): Dictionary {source_name: displayed_name}.
        displayed_units (dict of str to str): Dictionary {source_name: displayed_unit}.
    """
    # Use seaborn with the paper style
    sns.set_context("paper")
    sns.set_style("whitegrid")
    # For each source, channel and number of available sources, compute the RMSE, R2 and MAE
    # Store the results in a DataFrame with columns:
    # masked_source, channel, num_avail_sources, rmse, r2, mae

    # Compute the number of available sources in each sample.
    # A source S is available if S_time_delta is not NaT, and we don't count the
    # masked source, so we remove 1.
    results_df["num_avail_sources"] = results_df.apply(
        lambda x: sum([x[f"{s}_time_delta"] is not pd.NaT for s in all_sources]) - 1, axis=1
    )
    grouped = results_df.groupby(["masked_source", "channel", "num_avail_sources"])
    num_samples = grouped.size()
    rmse = grouped.apply(lambda x: np.sqrt(np.mean((x["target"] - x["pred"]) ** 2)))
    r2 = grouped.apply(lambda x: r2_score(x["target"], x["pred"]))
    mae = grouped.apply(lambda x: np.mean(np.abs(x["target"] - x["pred"])))
    # Group the results in a single DataFrame
    results_per_n_avail = pd.DataFrame(
        {
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "num_samples": num_samples,
        }
    ).reset_index()
    # Drop the groups with less than 10 samples to avoid meaningless metrics
    results_per_n_avail = results_per_n_avail[results_per_n_avail["num_samples"] >= 10]

    # Create one figure per (masked_source, channel) pair
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 4))
    axes = axes.ravel()

    for (msource, c), sub_df in results_per_n_avail.groupby(["masked_source", "channel"]):
        unit = f"({displayed_units[msource]})" if msource in displayed_units else ""
        displayed_name = displayed_names[msource] if msource in displayed_names else msource

        sns.lineplot(data=sub_df, x="num_avail_sources", y="rmse", markers=True, ax=axes[0])
        axes[0].set_title(f"RMSE - {displayed_name} / {c}")
        axes[0].set_ylabel(f"RMSE {unit}")

        sns.lineplot(data=sub_df, x="num_avail_sources", y="r2", markers=True, ax=axes[1])
        axes[1].set_title(f"R² - {displayed_name} / {c}")
        axes[1].set_ylabel("R²")

        sns.lineplot(data=sub_df, x="num_avail_sources", y="mae", markers=True, ax=axes[2])
        axes[2].set_title(f"MAE - {displayed_name} / {c}")
        axes[2].set_xlabel("Number of Available Sources")
        axes[2].set_ylabel(f"MAE {unit}")

        sns.barplot(
            data=sub_df,
            x="num_avail_sources",
            y="num_samples",
            ax=axes[3],
            native_scale=True,
            width=0.1,
        )
        axes[3].set_title(f"Samples - {displayed_name} / {c}")
        axes[3].set_ylabel("# Samples")

        x_vals = sorted(sub_df["num_avail_sources"].unique())
        axes[-1].set_xticks(x_vals)
        axes[-1].set_xticklabels([int(x) for x in x_vals])
        axes[-1].set_xlabel("Number of available Sources")

        for ax in axes[:2]:
            ax.tick_params(labelbottom=False)

        plt.tight_layout()
        plt.savefig(figures_dir / f"metrics_vs_num_sources_{msource}_{c}.png")
        plt.close()
