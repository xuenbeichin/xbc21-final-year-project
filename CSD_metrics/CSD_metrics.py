import numpy as np
from scipy.special import rel_entr
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score


import torch


class CSDMetricsCalculatorTorch:
    """
    Computes a variety of evaluation metrics between true and predicted Cross-Spectral Density (CSD) matrices.
    The inputs are assumed to be complex-valued and are processed using PyTorch tensors.

    Parameters:
    -----------
    real_csd : array-like or torch.Tensor
        The ground-truth cross-spectral density, provided as a complex array.

    predicted_csd : array-like or torch.Tensor
        The predicted cross-spectral density, also provided as a complex array.

    Metrics:
    --------
    - MSE (Mean Squared Error):
        Average squared difference between real and predicted CSD (real and imaginary parts).

    - NMSE (Normalised Mean Squared Error):
        Ratio of total squared error to the total power in the true CSD.

    - Relative Error:
        Norm of the error divided by the norm of the true CSD.

    - Explained Variance:
        Proportion of variance in the true CSD explained by the predicted CSD, based on magnitudes.

    - Pearson Correlation:
        Linear correlation coefficient between the real parts of the predicted and true CSD.

    - Cross-Correlation:
        Alias for Pearson correlation (computed on real parts).

    - Complex Coherence:
        Measures the squared magnitude of the cross-spectrum normalised by the product of the power spectra.

    - KL Divergence:
        Kullback-Leibler divergence between the normalised magnitudes of true and predicted CSD.

    Methods:
    --------
    - compute_mse()
    - compute_nmse()
    - compute_relative_error()
    - compute_explained_variance()
    - pearson_correlation()
    - compute_complex_coherence()
    - kl_divergence()
    - compute_cross_correlation()
    - evaluate(): Computes and prints all metrics, returning a summary dictionary.

    Returns:
    --------
    Dictionary containing all computed metrics in `evaluate()`.
    """
    def __init__(self, real_csd, predicted_csd):
        # Store inputs as complex tensors.
        self.real_csd = torch.as_tensor(real_csd, dtype=torch.complex64)
        self.predicted_csd = torch.as_tensor(predicted_csd, dtype=torch.complex64)

    def compute_mse(self):
        # Compute MSE over real and imaginary parts.
        mse = torch.mean((self.predicted_csd.real - self.real_csd.real) ** 2 +
                         (self.predicted_csd.imag - self.real_csd.imag) ** 2)
        return mse.item()

    def compute_nmse(self):
        # NMSE = sum of squared errors / sum of squared values of the true CSD.
        num = torch.sum((self.predicted_csd.real - self.real_csd.real) ** 2 +
                        (self.predicted_csd.imag - self.real_csd.imag) ** 2)
        den = torch.sum(self.real_csd.real ** 2 + self.real_csd.imag ** 2)
        return (num / den).item()

    def compute_relative_error(self):
        # Relative error computed as norm(error) / norm(true)
        error = torch.sqrt((self.predicted_csd.real - self.real_csd.real) ** 2 +
                           (self.predicted_csd.imag - self.real_csd.imag) ** 2)
        rel_error = torch.norm(error) / torch.norm(torch.sqrt(self.real_csd.real ** 2 + self.real_csd.imag ** 2))
        return rel_error.item()

    def compute_explained_variance(self):
        # Explained variance = 1 - (variance of error / variance of true CSD, computed on magnitudes)
        error = torch.sqrt((self.predicted_csd.real - self.real_csd.real) ** 2 +
                           (self.predicted_csd.imag - self.real_csd.imag) ** 2)
        variance_error = torch.var(error)
        true_mag = torch.sqrt(self.real_csd.real ** 2 + self.real_csd.imag ** 2)
        variance_true = torch.var(true_mag)
        explained_variance = 1 - (variance_error / variance_true)
        return explained_variance.item()

    def pearson_correlation(self):
        # For complex data, we'll compute Pearson correlation on the real part.
        x = self.real_csd.real.flatten()
        y = self.predicted_csd.real.flatten()
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
        corr = numerator / denominator if denominator > 0 else torch.tensor(0.0)
        return corr.item()

    def compute_complex_coherence(self):
        # Coherence = |S_xy|^2 / (S_xx * S_yy), computed on the complex signals.
        S_xy = torch.mean(self.real_csd * torch.conj(self.predicted_csd))
        S_xx = torch.mean(torch.abs(self.real_csd) ** 2)
        S_yy = torch.mean(torch.abs(self.predicted_csd) ** 2)
        coherence = (torch.abs(S_xy) ** 2) / (S_xx * S_yy)
        return coherence.item()

    def kl_divergence(self, epsilon=1e-8):
        # Use the magnitudes as distributions.
        true_mag = torch.abs(self.real_csd)
        pred_mag = torch.abs(self.predicted_csd)
        true_prob = true_mag / (torch.sum(true_mag) + epsilon)
        pred_prob = pred_mag / (torch.sum(pred_mag) + epsilon)
        kl_div = torch.sum(true_prob * torch.log((true_prob + epsilon) / (pred_prob + epsilon)))
        return kl_div.item()

    def compute_cross_correlation(self):
        # For simplicity, we can use Pearson correlation (on the real parts)
        return self.pearson_correlation()

    def evaluate(self):
        mse = self.compute_mse()
        nmse = self.compute_nmse()
        relative_error = self.compute_relative_error()
        explained_variance = self.compute_explained_variance()
        pearson_corr = self.pearson_correlation()
        coherence = self.compute_complex_coherence()
        kl_div = self.kl_divergence()
        cross_corr = self.compute_cross_correlation()

        print("=== Torch Optimisation Evaluation Results ===")
        print(f"MSE = {mse:.4e}, NMSE = {nmse:.4e}, Relative Error = {relative_error:.4e}")
        print(f"Explained Variance = {explained_variance:.4f}")
        print(f"Pearson Correlation = {pearson_corr:.4f}, Cross-Correlation = {cross_corr:.4f}")
        print(f"Coherence = {coherence:.4f}, KL Divergence = {kl_div:.4e}")

        return {
            "mse": mse,
            "nmse": nmse,
            "relative_error": relative_error,
            "explained_variance": explained_variance,
            "pearson_corr": pearson_corr,
            "cross_correlation": cross_corr,
            "coherence": coherence,
            "kl_divergence": kl_div,
        }

    def evaluate_parameters(self, true_params, predicted_params):
        """
        Evaluate parameter-level metrics for each noise parameter.
        Args:
            true_params (dict): Ground-truth values.
            predicted_params (dict): Recovered values with the same keys.
        Returns:
            dict: Metrics for each parameter.
        """
        metrics = {}
        for key in true_params:
            if key not in predicted_params:
                continue
            true_val = true_params[key]
            pred_val = predicted_params[key]
            # Convert to scalar if the value is a numpy array with one element.
            if isinstance(true_val, np.ndarray):
                true_val = true_val.item() if true_val.size == 1 else true_val
            if isinstance(pred_val, np.ndarray):
                pred_val = pred_val.item() if pred_val.size == 1 else pred_val

            abs_error = abs(true_val - pred_val)
            rel_error = abs_error / abs(true_val) if true_val != 0 else float('inf')
            mse = (true_val - pred_val) ** 2
            metrics[key] = {
                "absolute_error": abs_error,
                "relative_error": rel_error,
                "mse": mse
            }
        print("=== Parameter Evaluation Metrics ===")
        for param, vals in metrics.items():
            print(f"{param}: Absolute Error = {vals['absolute_error']:.4e}, "
                  f"Relative Error = {vals['relative_error']:.4e}, MSE = {vals['mse']:.4e}")
        return metrics


# --- Function to compute metrics for evaluation ---
class CSDMetricsCalculatorNp:
    def __init__(self, real_csd, predicted_csd):
        self.real_csd = real_csd
        self.predicted_csd = predicted_csd

    def compute_nmse(self):
        nmse = np.sum((self.predicted_csd - self.real_csd) ** 2) / np.sum(self.real_csd ** 2)
        return nmse

    def compute_mse(self):
        mse = np.mean((self.predicted_csd - self.real_csd) ** 2)
        return mse

    def compute_relative_error(self):
        relative_error = np.linalg.norm(self.predicted_csd - self.real_csd) / np.linalg.norm(self.real_csd)
        return relative_error

    def compute_explained_variance(self):
        explained_variance = 1 - (np.var(self.real_csd - self.predicted_csd) / np.var(self.real_csd))
        return explained_variance

    def pearson_correlation(self):
        # Compute Pearson correlation
        corr, _ = pearsonr(self.real_csd, self.predicted_csd)
        return corr

    def compute_complex_coherence(self):
        # Compute the complex coherence properly using cross-spectrum
        S_xy = np.mean(self.real_csd * np.conj(self.predicted_csd))
        S_xx = np.mean(np.abs(self.real_csd) ** 2)
        S_yy = np.mean(np.abs(self.predicted_csd) ** 2)
        coherence = np.abs(S_xy) ** 2 / (S_xx * S_yy)
        return coherence

    def kl_divergence(self):
        real_csd_prob = self.real_csd / np.sum(self.real_csd)
        predicted_csd_prob = self.predicted_csd / np.sum(self.predicted_csd)
        kl_divergence = np.sum(rel_entr(real_csd_prob, predicted_csd_prob))
        return kl_divergence

    def compute_mutual_information(self):
        mutual_info = mutual_info_score(self.real_csd.astype(int), self.predicted_csd.astype(int))
        return mutual_info

    def compute_cross_correlation(self):
        correlation_matrix = np.corrcoef(self.real_csd, self.predicted_csd)
        return correlation_matrix[0, 1]


    def evaluate(self):
        # Calculate all metrics
        mse = self.compute_mse()
        nmse = self.compute_nmse()
        relative_error = self.compute_relative_error()
        explained_variance = self.compute_explained_variance()
        corr = self.pearson_correlation()
        coherence = self.compute_complex_coherence()
        kl_div = self.kl_divergence()
        mutual_info = self.compute_mutual_information()
        cross_corr = self.compute_cross_correlation()

        print("=== Optimisation Evaluation Results ===")
        print(f"MSE = {mse:.4e}, NMSE = {nmse:.4e}, Relative Error = {relative_error:.4e}")
        print(f"Explained Variance = {explained_variance:.4f}, Mutual Info = {mutual_info:.4f}")
        print(f"Pearson Correlation = {corr:.4f}, Cross-Correlation = {cross_corr:.4f}")
        print(f"Mean Coherence = {coherence:.4f}, KL Divergence = {kl_div:.4f}")

        return {
            "mse": mse,
            "nmse": nmse,
            "relative_error": relative_error,
            "explained_variance": explained_variance,
            "pearson_corr": corr,
            "cross_correlation": cross_corr,
            "coherence": coherence,
            "kl_divergence": kl_div,
            "mutual_information": mutual_info,
        }