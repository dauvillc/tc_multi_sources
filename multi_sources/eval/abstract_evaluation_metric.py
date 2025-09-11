"""
Implements the AbstractEvaluationMetric class, which is a base class for all evaluation metrics.
"""

import abc
import re
from pathlib import Path

import xarray as xr


def models_info_sanity_check(info_dfs):
    """Performs a sanity check on the predictions info dataframes, to
    ensure they were made on exactly the same data."""
    # Check that all info_dfs have the same columns
    if not all(info_df.columns.equals(info_dfs[0].columns) for info_df in info_dfs):
        raise ValueError(
            "Found models with different columns in their info_df. "
            "Please ensure all models are evaluated on the same data."
        )
    # Check that they have the same number of rows (samples and sources)
    if not all(len(info_df) == len(info_dfs[0]) for info_df in info_dfs):
        raise ValueError(
            "Found models with different number of rows in their info_df. "
            "This indicates different samples or different sources. "
        )
    # Sort by sample_index, then by source name and then by source index
    for info_df in info_dfs:
        info_df.sort_values(by=["sample_index", "source_name", "source_index"], inplace=True)
        info_df.reset_index(drop=True, inplace=True)
    # Check that you the sample_index, source_name, source_index columns are the same
    for col in ["sample_index", "source_name", "source_index"]:
        if not all(info_df[col].equals(info_dfs[0][col]) for info_df in info_dfs):
            raise ValueError(
                f"Found models with different values in the {col} column of their info_df. "
                "Please ensure all models are evaluated on the same data."
            )
    # Check that are no duplicates in the (sample_index, source_name, source_index) triples
    for info_df in info_dfs:
        if info_df.duplicated(subset=["sample_index", "source_name", "source_index"]).any():
            raise ValueError(
                "Found duplicates in the (sample_index, source_name, source_index) triples "
                "of the info_df. There may be a bug, as a a pair (src_name, src_index) should "
                "be unique within a sample."
            )
    # Check the availability flags match
    for i in range(1, len(info_dfs)):
        if not info_dfs[i]["avail"].equals(info_dfs[0]["avail"]):
            raise ValueError(
                "Found models with different availability flags in their info_df. "
                "Make sure the models' masking selection is identical between"
                "the prediction runs."
            )


class AbstractMultisourceEvaluationMetric(abc.ABC):
    """Base class for all evaluation metrics."""

    def __init__(
        self, id_name, full_name, model_data, parent_results_dir, source_name_replacements=None
    ):
        """
        Args:
            id_name (str): Unique identifier for the metric. Must follow the
                rules of file naming.
            full_name (str): Full name of the metric, for display purposes.
            model_data (dict): Dictionary mapping model_ids to dictionaries containing:
                - info_df: DataFrame with metadata
                - root_dir: Path to predictions directory
                - results_dir: Path to results directory
                - run_id: Run ID
                - pred_name: Prediction name
            parent_results_dir (str or Path): Parent directory for all results
            source_name_replacements (List of tuple of str, optional): List of (pattern, replacement)
                substitutions to apply to source names for display purposes. The replacement
                is done using the re.sub function.
        """
        self.id_name = id_name
        self.full_name = full_name
        self.model_data = model_data
        self.parent_results_dir = Path(parent_results_dir)
        self.source_name_replacements = source_name_replacements or []

        # Create a directory for this evaluation metric
        self.metric_results_dir = self.parent_results_dir / id_name
        self.metric_results_dir.mkdir(parents=True, exist_ok=True)

        # Perform sanity checks on the model data (also sorts the dataframes)
        info_dfs = [model_spec["info_df"] for model_spec in model_data.values()]
        models_info_sanity_check(info_dfs)

        # Isolate a model-agnostic DataFrame with the columns that don't depend on the model:
        # sample_index, source_name, source_index, avail, dt
        self.samples_df = (
            info_dfs[0][["sample_index", "source_name", "source_index", "avail", "dt"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.n_samples = self.samples_df["sample_index"].nunique()

    def samples_iterator(self):
        """Iterator over the samples in the evaluation.
        Yields:
            pandas.DataFrame: A DataFrame with the columns:
                - sample_index: Index of the sample (same for all rows)
                - source_name: Name of the source
                - source_index: Index of the source
                - avail: Availability flag (1 for available, 0 for target)
                - dt: Timestamp of the sample
                only the data for a single sample (sample_index) is yielded at a time.
            dict: A dictionary with the keys:
                - targets: Dict (source_name, source_index) -> xarray.Dataset
                    The targets are the same for all models.
                - predictions: Dict model_id -> Dict (source_name, source_index) -> xarray.Dataset
                - embeddings: Dict model_id -> Dict (source_name, source_index) -> xarray.Dataset
                    Only available for models that return their embeddings.
        The missing (source_name, source_index) pairs, i.e. those with an availability flag
        of -1, are removed from both the DataFrame and xarray datasets.
        """
        for sample_index in self.samples_df["sample_index"].unique():
            targets, predictions, embeddings = self.load_data(sample_index)
            sample_df = self.samples_df[self.samples_df["sample_index"] == sample_index]
            sample_df = sample_df.set_index(["source_name", "source_index"])

            sample_data = {
                "targets": {},
                "predictions": {model_id: {} for model_id in self.model_data},
                "embeddings": {model_id: {} for model_id in self.model_data},
            }
            for src in sample_df.index:
                # Get the availability flag to know whether this src is available
                avail = sample_df.loc[src, "avail"]
                if avail == -1:
                    continue
                sample_data["targets"][src] = targets[src]
                for model_id in self.model_data:
                    sample_data["predictions"][model_id][src] = predictions[model_id][src]
                    if model_id in embeddings and src in embeddings[model_id]:
                        sample_data["embeddings"][model_id][src] = embeddings[model_id][src]

            # Remove the rows from the DataFrame that are not available
            sample_df = sample_df[sample_df["avail"] != -1]
            yield sample_df, sample_data

    def load_data(self, sample_index):
        """Loads the data for all models as xarray datasets stored in individual
        netCDF4 files.
        Args:
            sample_index (int): Index of the sample to load.
        Returns:
            targets (dict): Dict (src_name, src_index) -> xarray.Dataset
               The targets are the same for all models.
            predictions (dict): Dict model_id -> (src_name, src_index) -> xarray.Dataset
            embeddings (dict): Dict model_id -> (src_name, src_index) -> xarray.Dataset
                Only if embeddings are available.
        """
        targets, predictions, embeddings = {}, {}, {}
        for i, (model_id, model_spec) in enumerate(self.model_data.items()):
            predictions[model_id], embeddings[model_id] = {}, {}
            info_df = model_spec["info_df"]
            info_df = info_df[info_df["sample_index"] == sample_index]
            root_dir = model_spec["root_dir"]

            # Isolate the unique pairs (source_name, source_index) for this model
            source_pairs = info_df[["source_name", "source_index"]].drop_duplicates()
            for _, row in source_pairs.iterrows():
                src, index = row["source_name"], row["source_index"]
                # Load targets for the first model only (since they are the same for all)
                if i == 0:
                    target_path = root_dir / "targets" / src / str(index) / f"{sample_index}.nc"
                    targets[(src, index)] = xr.open_dataset(target_path)
                # Load predictions for all models
                pred_path = root_dir / "predictions" / src / str(index) / f"{sample_index}.nc"
                predictions[model_id][(src, index)] = xr.open_dataset(pred_path)
                # Load embeddings if available
                emb_path = root_dir / "embeddings" / src / str(index) / f"{sample_index}.nc"
                if emb_path.exists():
                    embeddings[model_id][(src, index)] = xr.open_dataset(emb_path)

        return targets, predictions, embeddings

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        """
        Main evaluation method.

        Args:
            **kwargs: Additional keyword arguments including:
                - num_workers: Number of workers for parallel processing
        """
        pass

    def _display_src_name(self, src_name):
        """Applies the source name replacements to a source name for display purposes."""
        for pattern, replacement in self.source_name_replacements:
            src_name = re.sub(pattern, replacement, src_name)
        return src_name
