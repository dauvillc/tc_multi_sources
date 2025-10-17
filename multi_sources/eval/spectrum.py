"""Implements the QuantativeEvaluation class."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


class RadiallyAveragedPSDEvaluation(AbstractMultisourceEvaluationMetric):
    """Evaluation class that computes the radially averaged power spectral density (PSD)
    for each model's predictions and the target data, and compares them in plots.
    """

    def __init__(self, model_data, parent_results_dir, **kwargs):
        """
        Args:
            model_data (dict): Dictionary mapping model_ids to model specifications.
            parent_results_dir (str or Path): Parent directory for all results.
            **kwargs: Additional keyword arguments passed to the AbstractMultisourceEvaluationMetric.
        """
        super().__init__(
            id_name="spectrum",
            full_name="Radially Averaged Power Spectral Density Evaluation",
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
                Includes the columns 'model_id', 'sample_index', 'source_name', 'source_index',
                'channel', 'mae', 'mse', 'crps', 'ssr'.
        """
        results = []  # List of dictionaries that we'll concatenate later into a DataFrame
        for sample_df, sample_data in tqdm(
            self.samples_iterator(), desc="Evaluating samples", total=self.n_samples
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
                    for channel in channels:
                        pred_data_channel = pred_data[channel].values
                        target_data_channel = target_data[channel].values
                        # For now, we'll skip the minority of cases where the target data
                        # contains NaNs.
                        if not np.isnan(target_data_channel).any():
                            # If there isn't a realization dimension, add one for consistency
                            if pred_data_channel.ndim == target_data_channel.ndim:
                                pred_data_channel = np.expand_dims(pred_data_channel, axis=0)
                            # Compute the PSD gain
                            psd_gain, freq = self._compute_psd_gain(
                                pred_data_channel, target_data_channel
                            )
                            sample_results_dict = {
                                "model_id": model_id,
                                "sample_index": sample_index,
                                "source_name": source_name,
                                "source_index": source_index,
                                "channel": channel,
                                "psd_gain": list(psd_gain),
                                "freq": list(freq),
                            }
                            results.append(sample_results_dict)

        # Concatenate all results into a single DataFrame
        return pd.DataFrame(results).explode(column=["psd_gain", "freq"]).reset_index(drop=True)

    def _plot_results(self, results):
        """Generates and saves plots comparing the models' PSD gains.

        Args:
            results (pd.DataFrame): DataFrame containing the evaluation results.
        """
        sns.set_theme(style="whitegrid")
        # We'll plot the average PSD gain over all samples for each model
        # on the same plot for comparison, for each channel.
        channels = results["channel"].unique()
        for channel in channels:
            plt.figure(figsize=(10, 6))
            channel_results = results[results["channel"] == channel]
            sns.lineplot(
                data=channel_results,
                x="freq",
                y="psd_gain",
                hue="model_id",
                estimator="mean",
                errorbar="sd",
            )
            plt.title(f"Radially Averaged PSD Gain - Channel: {channel}")
            plt.xlabel("Frequency")
            plt.ylabel("PSD Gain")
            plt.legend(title="Model ID")
            plt.xscale("log")
            plt.yscale("log")
            plt.grid(True, which="both", ls="--", lw=0.5)
            plot_file = self.metric_results_dir / f"psd_gain_{channel}.png"
            plt.savefig(plot_file)
            plt.close()
            print(f"Plot saved to: {plot_file}")

    @staticmethod
    def _compute_psd_gain(pred_data, target_data):
        """
        Compute the radially averaged power spectral density (RAPSD) gain between
        the prediction and target data.

        Args:
            pred_data (ndarray): Predicted data, of shape (M, H, W) where M is the number of
                realizations. The PSD gain is averaged over the M realizations.
            target_data (ndarray): Target data, of shape (H, W) matching the shape of each
                realization in pred_data.
        Returns:
            psd_gain (ndarray): The RAPSD gain, of shape (L,) where L is the number of
                frequency bins.
            freq (ndarray): The corresponding frequencies, of shape (L,).
        """
        # Remove the mean to avoid a spike at the zero frequency
        pred_data = pred_data - np.mean(pred_data, axis=(-2, -1), keepdims=True)
        target_data = target_data - np.mean(target_data)

        # Compute the RAPSD for each realization and average over realizations
        for i in range(pred_data.shape[0]):
            pred_psd, freq = rapsd(pred_data[i], fft_method=np.fft, return_freq=True)
            if i == 0:
                pred_psd_avg = pred_psd
            else:
                pred_psd_avg += pred_psd
        pred_psd_avg /= pred_data.shape[0]
        target_psd, _ = rapsd(target_data, fft_method=np.fft, return_freq=True)

        psd_gain = pred_psd / (target_psd + 1e-10)

        return psd_gain, freq


def compute_centred_coord_array(M, N):
    """
    Compute a 2D coordinate array, where the origin is at the center.
    Taken from the pysteps package.

    M : int
      The height of the array.
    N : int
      The width of the array.

    Returns
    -------
    out : ndarray
      The coordinate array.

    Examples
    --------
    >>> compute_centred_coord_array(2, 2)

    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))

    """

    if M % 2 == 1:
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC


def rapsd(field, fft_method=None, return_freq=False, d=1.0, normalize=False, **fft_kwargs):
    """
    Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.
    Taken from the pysteps package.

    Args:
        field (ndarray): A 2d array of shape (m, n) containing the input field.
        fft_method (Callable): A module or object implementing the same methods as numpy.fft and
            scipy.fftpack. If set to None, field is assumed to represent the
            shifted discrete Fourier transform of the input field, where the
            origin is at the center of the array
            (see numpy.fft.fftshift or scipy.fftpack.fftshift).
        return_freq (bool): Whether to also return the Fourier frequencies.
        d (scalar): Sample spacing (inverse of the sampling rate). Defaults to 1.
            Applicable if return_freq is 'True'.
        normalize (bool): If True, normalize the power spectrum so that it sums to one.
            Whether to also return the Fourier frequencies.

    Returns:
        out (ndarray): One-dimensional array containing the RAPSD. The length of the array is
            int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
        freq (ndarray): One-dimensional array containing the Fourier frequencies.
    """

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number " "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field, **fft_kwargs))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result
