"""
Implements the AbstractEvaluationMetric class, which is a base class for all evaluation metrics.
"""
import numpy as np
import abc
from pathlib import Path


class AbstractEvaluationMetric(abc.ABC):
    """Base class for all evaluation metrics."""
    def __init__(self, id_name, full_name, predictions_dir, results_dir):
        """
        Args:
            id_name (str): Unique identifier for the metric. Must follow the
                rules of file naming.
            full_name (str): Full name of the metric, for display purposes.
            predictions_dir (str or Path): Directory in which the predictions
                and targets are stored.
            results_dir (str or Path): Directory in which the results from all
                metrics will be written.
        """
        self.id_name = id_name
        self.full_name = full_name
        self.predictions_dir = Path(predictions_dir)
        self.results_dir = Path(results_dir) / id_name
        # Create the directory if it does not exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_batch(self, source_name, batch_idx):
        """
        Loads a batch (targets and predictions) for a given source.

        Args:
            source_name (str): Name of the source.
            batch_idx (int): Index of the batch.
        Returns:
            targets (np.ndarray or None): Array of shape (batch_size, 2 + channels, height, width),
                or None if the source was not included in the batch. The first two channels are
                the latitude and longitude at each pixel.
            predictions (np.ndarray or None): Array of shape (batch_size, channels, height, width),
                or None if the source was not included in the predictions.
        """
        targets_path = self.predictions_dir / "targets" / source_name / f"{batch_idx}.npy"
        predictions_path = self.predictions_dir / "outputs" / source_name / f"{batch_idx}.npy"
        if not targets_path.exists():
            return None, None
        targets = np.load(targets_path)
        if not predictions_path.exists():
            return targets, None
        predictions = np.load(predictions_path)
        return targets, predictions


class AbstractSourcewiseEvaluationMetric(AbstractEvaluationMetric):
    """Base class for evaluation metrics that process each source separately."""
    @abc.abstractmethod
    def evaluate_source(source_name, info_df, **kwargs):
        """
        Processes the data for a given source.
        
        Args:
            source_name (str): Name of the source.
            info_df (pd.DataFrame): DataFrame with at least the following columns:
                avail, batch_idx, index_in_batch, dt.
            **kwargs: Additional keyword arguments.
        """
        pass

    def create_source_results_dir(self, source_name):
        """Creates the directory for the results of a given source."""
        source_dir = self.results_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        return source_dir


class AbstractMultisourceEvaluationMetric(AbstractEvaluationMetric):
    """Base class for evaluation metrics that process all sources together."""
    @abc.abstractmethod
    def evaluate_sources(info_df, **kwargs):
        """
        Processes the data for all sources.
        
        Args:
            info_df (pd.DataFrame): DataFrame with at least the following columns:
                source_name, avail, batch_idx, index_in_batch, dt.
            **kwargs: Additional keyword arguments.
        """
        pass