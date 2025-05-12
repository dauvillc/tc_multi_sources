"""
Implements the AbstractEvaluationMetric class, which is a base class for all evaluation metrics.
"""

import abc
from pathlib import Path

import xarray as xr


class AbstractEvaluationMetric(abc.ABC):
    """Base class for all evaluation metrics."""

    def __init__(self, id_name, full_name, model_data, parent_results_dir):
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
        """
        self.id_name = id_name
        self.full_name = full_name
        self.model_data = model_data
        self.parent_results_dir = Path(parent_results_dir)

        # Create a directory for this evaluation metric
        self.metric_results_dir = self.parent_results_dir / id_name
        self.metric_results_dir.mkdir(parents=True, exist_ok=True)

        # Store directories for each model
        self.model_dirs = {}
        for model_id, data in model_data.items():
            model_results_dir = self.metric_results_dir / model_id

            # Create or reset the directory
            if model_results_dir.exists():
                for item in model_results_dir.iterdir():
                    if item.is_file():
                        item.unlink()
            else:
                model_results_dir.mkdir(parents=True, exist_ok=True)

            self.model_dirs[model_id] = {
                "predictions_dir": data["root_dir"],
                "results_dir": model_results_dir,
            }

    def load_batch(self, source_name, index, batch_idx, model_id):
        """
        Loads a batch (targets and predictions) for a given source at a specific index.

        Args:
            source_name (str): Name of the source.
            index (int): Index of the observation (0 = most recent).
            batch_idx (int): Index of the batch.
            model_id (str): Identifier for the model (run_id_pred_name)
        Returns:
            targets (xarray.Dataset or None): The targets dataset, or None if not available.
            predictions (xarray.Dataset or None): The predictions dataset, or None if not available.
        """
        predictions_dir = self.model_dirs[model_id]["predictions_dir"]
        targets_path = predictions_dir / "targets" / source_name / str(index) / f"{batch_idx}.nc"
        predictions_path = (
            predictions_dir / "outputs" / source_name / str(index) / f"{batch_idx}.nc"
        )
        if not targets_path.exists():
            return None, None
        targets = xr.open_dataset(targets_path)
        if not predictions_path.exists():
            return targets, None
        predictions = xr.open_dataset(predictions_path)

        return targets, predictions


class AbstractSourcewiseEvaluationMetric(AbstractEvaluationMetric):
    """Base class for evaluation metrics that process each source separately."""

    @abc.abstractmethod
    def evaluate_source(self, source_name, info_df, **kwargs):
        """
        Processes the data for a given source.

        Args:
            source_name (str): Name of the source.
            info_df (pd.DataFrame): DataFrame with at least the following columns:
                avail, batch_idx, index_in_batch, dt.
            **kwargs: Additional keyword arguments including:
                - run_id: Run ID
                - pred_name: Prediction name
                - model_id: Model identifier
        """
        pass

    def create_source_results_dir(self, source_name, model_id, index=None):
        """Creates the directory for the results of a given source at a specific index.

        Args:
            source_name (str): Name of the source.
            model_id (str): Identifier for the model (run_id_pred_name)
            index (optional): Index of the observation (0 = most recent). If None, index is not included in the path.

        Returns:
            Path: Path to the results directory.
        """
        model_results_dir = self.model_dirs[model_id]["results_dir"]

        if index is None:
            source_dir = model_results_dir / source_name
        else:
            source_dir = model_results_dir / source_name / str(index)
        source_dir.mkdir(parents=True, exist_ok=True)
        return source_dir


class AbstractMultisourceEvaluationMetric(AbstractEvaluationMetric):
    """Base class for evaluation metrics that process all sources together."""

    @abc.abstractmethod
    def evaluate_sources(self, **kwargs):
        """
        Processes the data for all sources across all models.

        Args:
            **kwargs: Additional keyword arguments including:
                - num_workers: Number of workers for parallel processing
        """
        pass
