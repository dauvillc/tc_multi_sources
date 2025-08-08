"""Implements the EmbeddingsComparisonEvaluation class."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cosine
from tqdm import tqdm

from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


class EmbeddingsComparisonEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluation class that, for each model separately, computes the similarity between
    the embeddings of the available non-masked sources and the embeddings of the
    masked sources. The similarity is computed using the cosine similarity.

    This evaluation expects all included models to return their values, coordinates and
    conditioning embeddings in their predict_step method.
    The per-sample metrics are saved to disk in a JSON file.
    """

    def __init__(self, model_data, parent_results_dir):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
        """
        super().__init__(
            id_name="embeddings_comparison",
            full_name="Embeddings Comparison Evaluation",
            model_data=model_data,
            parent_results_dir=parent_results_dir,
        )

    def evaluate(self, **kwargs):
        """Main evaluation method that processes the data for all models."""

        # Evaluate all models and save the results
        results = self._evaluate_embeddings_similarities()
        results_file = self.metric_results_dir / "similarities.json"
        results.to_json(results_file, orient="records", lines=True)
        print(f"Embeddings similarities saved to: {results_file}")

        # Generate and save plots comparing the models
        self._plot_embeddings_similarities(results)

        # Evaluate the error metrics against the embeddings similarities
        self._evaluate_error_vs_embeddings_similarities(results)
        return

    def _evaluate_embeddings_similarities(self):
        """Evaluates the similarities between embeddings for all models.
        Returns:
            results (pd.DataFrame): DataFrame containing the evaluation results for the model.
                Includes the columns 'model_id', 'sample_index', 'source_name', 'source_index',
                'target_source_name', 'target_source_index', 'channel',
                'coords_similarity', 'cond_similarity'.
        """
        results = []  # List of dictionaries that we'll concatenate later into a DataFrame.
        for sample_df, sample_data in tqdm(
            self.samples_iterator(),
            desc="Evaluating embeddings similarities",
            total=self.n_samples,
        ):
            sample_index = sample_df["sample_index"].iloc[0]
            # For each target source, compute the similarity with each available, non-target
            # source. To do so, we'll first retrieve the availability flags of all sources.
            avail_flags = sample_df["avail"].to_dict()
            # Iterate over all target sources (i.e. sources with availability flag 0).
            for target_src in avail_flags:
                target_source_name, target_source_index = target_src
                if avail_flags[target_src] != 0:
                    continue
                # Inner loop: iterate over all non-target, available sources.
                for src in avail_flags:
                    source_name, source_index = src
                    if avail_flags[src] != 1:
                        continue
                    # Inner inner loop: iterate over all models.
                    for model_id in self.model_data:
                        embed_data = sample_data["embeddings"][model_id]
                        target_coords_embeddings = embed_data[target_src]["coords"].values
                        target_cond_embeddings = embed_data[target_src]["conditioning"].values
                        src_coords_embeddings = embed_data[src]["coords"].values
                        src_cond_embeddings = embed_data[src]["conditioning"].values

                        # Compute the cosine similarity between the embeddings.
                        coords_similarity = self._compute_cosine_similarity(
                            target_coords_embeddings, src_coords_embeddings
                        )
                        cond_similarity = self._compute_cosine_similarity(
                            target_cond_embeddings, src_cond_embeddings
                        )
                        results.append(
                            {
                                "model_id": model_id,
                                "sample_index": sample_index,
                                "source_name": source_name,
                                "source_index": source_index,
                                "target_source_name": target_source_name,
                                "target_source_index": target_source_index,
                                "coords_similarity": coords_similarity,
                                "cond_similarity": cond_similarity,
                            }
                        )
        # Convert the results to a DataFrame.
        results_df = pd.DataFrame(results)
        return results_df

    @staticmethod
    def _compute_cosine_similarity(embed1, embed2):
        """Computes the cosine similarity between two embeddings.
        The embeddings are expected to be of shape (..., n_features),
        where ... represents any number of leading dimensions, which can differ
        between the two embeddings. The embeddings are averaged over those leading
        dimensions before computing the cosine similarity.

        Args:
            embed1 (np.ndarray): First embedding.
            embed2 (np.ndarray): Second embedding.

        Returns:
            float: Cosine similarity between the two embeddings.
        """
        # Average over all dimensions except the last one.
        embed1_avg = embed1.mean(axis=tuple(range(embed1.ndim - 1)))
        embed2_avg = embed2.mean(axis=tuple(range(embed2.ndim - 1)))
        # Flatten the averaged embeddings to 1D arrays.
        embed1_avg = embed1_avg.flatten()
        embed2_avg = embed2_avg.flatten()
        # Compute the cosine similarity.
        similarity = 1 - cosine(embed1_avg, embed2_avg)
        return similarity

    def _plot_embeddings_similarities(self, results):
        """Generates a plot showing the joint distribution of the coordinates and conditioning
        embeddings similarities for each model.
        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results.
        """
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=results,
            x="coords_similarity",
            y="cond_similarity",
            hue="model_id",
            style="model_id",
            alpha=0.7,
        )
        plt.title("Embeddings Similarity Comparison")
        plt.xlabel("Coordinates Similarity")
        plt.ylabel("Conditioning Similarity")
        plt.legend(title="Model ID")
        plt.tight_layout()
        plot_file = self.metric_results_dir / "embeddings_similarity_plot.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Embeddings similarity plot saved to: {plot_file}")

    def _evaluate_error_vs_embeddings_similarities(self, embeddings_results):
        """Evaluates the MSE against the embeddings similarities for all models.

        Since there are multiple embeddings similarities in each sample (one per
        (source, target source) pair), we'll use the maximum similarity for each
        (sample, source_name, source_index) triplet: for each target source T, we'll
        look at either the maximum or the average similarity to T
        across all available sources S.

        Assumes that the MSE per-sample have already been computed using the
        QuantitativeEvaluation class.
        - Generates a figure showing the joint distribution of the MSE against the coordinates
        embeddings similarities.
        - Generates the same figures but with the conditioning embeddings similarities.
        Args:
            embeddings_results (pd.DataFrame): DataFrame containing the embeddings similarities results.
        """
        # Look for the quantitative evaluation results in the parent directory.
        quantitative_results_file = self.parent_results_dir / "quantitative" / "full_results.json"
        if not quantitative_results_file.exists():
            # Skip this evaluation if the quantitative results are not available.
            print(
                "Quantitative results not found, skipping error vs embeddings similarities evaluation."
            )
            return
        print(f"Loading quantitative results from: {quantitative_results_file}")
        quantitative_results = pd.read_json(
            quantitative_results_file, orient="records", lines=True
        )

        # Compute the highest and mean embeddings similarities for each sample and target source
        aggregated_similarities = (
            embeddings_results.groupby(
                ["sample_index", "target_source_name", "target_source_index"]
            )["coords_similarity", "cond_similarity"]
            .agg(["max", "mean"])
            .reset_index()
        )
        # Rename the columns to match the error results dataframe.
        aggregated_similarities.columns = [
            "sample_index",
            "source_name",  # target_source_name
            "source_index",  # target_source_index
            "coords_similarity_max",
            "coords_similarity_mean",
            "cond_similarity_max",
            "cond_similarity_mean",
        ]
        # Sanity check: the aggregated similarities should have the same number
        # of rows as the quantitative results.
        assert len(aggregated_similarities) == len(quantitative_results), (
            "The number of samples in the aggregated similarities results does not match "
            "the number of samples in the quantitative results."
        )
        # Merge the quantitative results with the aggregated similarities.
        merged_results = pd.merge(
            quantitative_results,
            aggregated_similarities,
            on=["sample_index", "source_name", "source_index"],
            how="left",
        )

        # Plotting.
        sns.set_theme(style="whitegrid")
        for model_id in self.model_data:
            model_results = merged_results[merged_results["model_id"] == model_id]
            # For Coordinates Similarity: create two subplots side by side,
            # for max and mean similarities.
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sns.regplot(
                data=model_results,
                x="coords_similarity_max",
                y="mse",
                ax=axes[0],
                order=2,  # Polynomial regression of order 2
                marker="x",
                color=".3",
                line_kws={"color": "red"},
            )
            axes[0].set_title("MSE vs Max Coordinates Similarity - Model: " + model_id)
            axes[0].set_xlabel("Max Coordinates Similarity")
            axes[0].set_ylabel("Mean Squared Error (MSE)")
            axes[0].set_xlim(0.9, 1.0)
            sns.regplot(
                data=model_results,
                x="coords_similarity_mean",
                y="mse",
                ax=axes[1],
                order=2,  # Polynomial regression of order 2
                marker="x",
                color=".3",
                line_kws={"color": "red"},
            )
            axes[1].set_title("MSE vs Mean Coordinates Similarity - Model: " + model_id)
            axes[1].set_xlabel("Mean Coordinates Similarity")
            axes[1].set_ylabel("Mean Squared Error (MSE)")
            axes[1].set_xlim(0.9, 1.0)
            sns.despine()
            plt.tight_layout()
            plot_file = self.metric_results_dir / f"mse_vs_coords_similarity_{model_id}.png"
            plt.savefig(plot_file)
            plt.close()
            print(f"MSE vs Coordinates Similarity plot for {model_id} saved to: {plot_file}")
