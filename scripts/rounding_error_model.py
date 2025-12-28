# rounding error model
# author: Sahil Bhola, University of Michigan, 2025

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

plt.style.use("./journal.mplstyle")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def empirical_cdf(samples):
    """
    Compute the empirical CDF from 1D samples.

    Args:
        samples: 1D array-like of samples

    Returns:
        cdf_func: A callable function that returns CDF values for given x values
        x_sorted: Sorted sample values (for reference)
        cdf_values: Corresponding CDF values at each sorted sample
    """
    samples = np.asarray(samples).flatten()
    n = len(samples)

    # Sort samples
    x_sorted = np.sort(samples)

    # Compute empirical CDF values (0 to 1)
    cdf_values = np.arange(1, n + 1) / n

    # Create interpolation function
    # For values outside the range, use step function behavior
    cdf_func = interpolate.interp1d(
        x_sorted, cdf_values, kind="next", fill_value=(0, 1), bounds_error=False
    )

    return cdf_func, x_sorted, cdf_values


class IEEEModel:
    def __init__(self, model):
        self.model = model
        assert self.model in ["double", "single", "half"], "invalid model"
        self.precision = self._get_precision()
        self.base = 2
        self.exponent_range = self._get_exponent_range()
        self.urd = 0.5 * self.base ** (1.0 - self.precision)

    def _get_precision(self):
        precision = {"double": 53, "single": 24, "half": 11}
        return precision.get(self.model)

    def _get_exponent_range(self):
        exponent_range = {
            "double": [-1021, 1024],
            "single": [-125, 128],
            "half": [-13, 16],
        }
        return exponent_range.get(self.model)


class uniform_model:
    """
    delta ~ U[-urd, urd]
    """

    def __init__(self, model="single"):

        self.model = IEEEModel(model)
        self._plot_pdf(n_samples=50000)  # probability density function
        self._plot_cdf(n_samples=50000)  # cummulative distribution function

        # stats
        self.delta_stats = self._get_delta_statistics()
        self.delta_emperical_stats = self._get_delta_emperical_statistics()

        self.log1pdelta_stats = self._get_log1pdelta_statistics()
        self.log1pdelta_emperical_stats = self._get_log1pdelta_emperical_statistics()

        print("==" * 10 + " Uniform Model " + "==" * 10)
        print(
            "[Delta stats] :"
            + ", ".join(f"{k}: {v:.3e}" for k, v in self.delta_stats.items())
        )
        print(
            "[Delta emperical stats] :"
            + ", ".join(f"{k}: {v:.3e}" for k, v in self.delta_emperical_stats.items())
        )
        print(
            "[Log(1+Delta) stats] :"
            + ", ".join(f"{k}: {v:.3e}" for k, v in self.log1pdelta_stats.items())
        )
        print(
            "[Log(1+Delta) emperical stats] :"
            + ", ".join(
                f"{k}: {v:.3e}" for k, v in self.log1pdelta_emperical_stats.items()
            )
        )

    def _get_delta_statistics(self):
        """statistics of rounding error delta"""
        a = -self.model.urd
        b = self.model.urd
        mean = 0.5 * (a + b)  # mean of uniform distribution
        var = (1.0 / 12.0) * (b - a) ** 2  # variance of uniform distribution
        bound = max(abs(a), abs(b))  # bound of uniform distribution

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _get_log1pdelta_statistics(self):
        """statistics of log(1 + delta), with delta~U[-urd, urd]"""
        u = self.model.urd
        kappa = -1.0 + u**2
        logm = np.log(1 - u)
        logp = np.log(1 + u)
        mean = (-2 * u + (-1 + u) * logm + (1 + u) * logp) / (2 * u)
        var = (
            4 * u**2 + kappa * logm**2 - 2 * kappa * logm * logp + kappa * logp**2
        ) / (4 * u**2)
        bound = logp

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _get_delta_emperical_statistics(self, n_samples: int = 100000):
        # delta
        delta = np.random.uniform(-self.model.urd, self.model.urd, n_samples)

        mean = np.mean(delta)
        var = np.var(delta)
        bound = np.max(np.abs(delta))

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _get_log1pdelta_emperical_statistics(self, n_samples: int = 100000):
        # delta
        delta = np.random.uniform(-self.model.urd, self.model.urd, n_samples)
        # log(1 + delta)
        log1pdelta = np.log(1 + delta)
        mean = np.mean(log1pdelta)
        var = np.var(log1pdelta)
        bound = np.max(np.abs(log1pdelta))

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _plot_pdf(self, n_samples: int = 100000):
        # delta
        delta = np.random.uniform(-self.model.urd, self.model.urd, n_samples)
        # delta_stats = self._get_delta_statistics()
        # log(1 + delta)
        log1pdelta = np.log(1 + delta)
        # log1pdelta_stats = self._get_log1pdelta_statistics()

        fig, axs = plt.subplots(
            1, 2, figsize=(10, 3), sharey=True, sharex=True, layout="compressed"
        )
        axs[0].hist(delta, density=True, bins=50)
        axs[0].set_xlabel(r"$\delta$")
        axs[0].set_ylabel(r"$f_{\delta}(\delta)$")
        axs[1].hist(log1pdelta, density=True, bins=50)
        axs[1].set_xlabel(r"$\log(1 + \delta)$")
        axs[1].set_ylabel(r"$f_{\log(1+\delta)}(\log(1+\delta)$")
        plt.savefig("uniform_delta_model_pdf.png")
        plt.close()

    def _plot_cdf(self, n_samples: int = 100000):
        # delta
        delta = np.random.uniform(-self.model.urd, self.model.urd, n_samples)
        cdf_func_delta, delta_sorted, cdf_delta = empirical_cdf(delta)
        # log(1 + delta)
        log1pdelta = np.log(1 + delta)
        cdf_func_log1pdelta, log1pdelta_sorted, cdf_log1pdelta = empirical_cdf(
            log1pdelta
        )

        fig, axs = plt.subplots(
            1, 2, figsize=(10, 4), sharey=True, sharex=True, layout="compressed"
        )
        axs[0].plot(delta_sorted, cdf_delta)
        axs[0].set_xlabel(r"$\delta$")
        axs[0].set_ylabel(r"$F_{\delta}(\delta)$")
        axs[1].plot(log1pdelta_sorted, cdf_log1pdelta)
        axs[1].set_xlabel(r"$\log(1 + \delta)$")
        axs[1].set_ylabel(r"$F_{\log(1+\delta)}(\log(1+\delta)$")
        plt.savefig("uniform_delta_model_cdf.png")
        plt.close()


class uniform_log_model:
    """
    log(1+delta) ~ U[log(1-urd), log(1+urd)]
    """

    def __init__(self, model="single"):
        self.model = IEEEModel(model)
        self._plot_pdf(n_samples=50000)  # probability density function
        self._plot_cdf(n_samples=50000)  # cummulative distribution function

        # stats

        self.delta_stats = self._get_delta_statistics()
        self.delta_emperical_stats = self._get_delta_emperical_statistics()

        self.log1pdelta_stats = self._get_log1pdelta_statistics()
        self.log1pdelta_emperical_stats = self._get_log1pdelta_emperical_statistics()

        print("==" * 10 + " Log Uniform Model " + "==" * 10)
        print(
            "[Delta stats] :"
            + ", ".join(f"{k}: {v:.3e}" for k, v in self.delta_stats.items())
        )
        print(
            "[Delta emperical stats] :"
            + ", ".join(f"{k}: {v:.3e}" for k, v in self.delta_emperical_stats.items())
        )
        print(
            "[Log(1+Delta) stats] :"
            + ", ".join(f"{k}: {v:.3e}" for k, v in self.log1pdelta_stats.items())
        )
        print(
            "[Log(1+Delta) emperical stats] :"
            + ", ".join(
                f"{k}: {v:.3e}" for k, v in self.log1pdelta_emperical_stats.items()
            )
        )

    def _get_delta_statistics(self):
        """statistics of rounding error delta"""
        u = self.model.urd
        L = np.log((1.0 - u) / (1.0 + u))
        D = u - np.arctanh(u)

        # mean
        mean = -2.0 * D / L

        # variance
        var = 2.0 * D * ((1.0 / L) - (2.0 * D / L**2))

        # bound
        bound = u

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _get_log1pdelta_statistics(self):
        """statistics of log(1 + delta), with delta~U[-urd, urd]"""
        a = np.log(1 - self.model.urd)
        b = np.log(1 + self.model.urd)
        mean = 0.5 * (a + b)  # mean of uniform distribution
        var = (1.0 / 12.0) * (b - a) ** 2  # variance of uniform distribution
        bound = max(abs(a), abs(b))  # bound of uniform distribution

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _get_delta_emperical_statistics(self, n_samples: int = 100000):
        # log(1 + delta)
        log1pdelta = np.random.uniform(
            np.log(1 - self.model.urd), np.log(1 + self.model.urd), n_samples
        )
        # delta
        delta = np.exp(log1pdelta) - 1

        mean = np.mean(delta)
        var = np.var(delta)
        bound = np.max(np.abs(delta))

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _get_log1pdelta_emperical_statistics(self, n_samples: int = 100000):
        # log(1 + delta)
        log1pdelta = np.random.uniform(
            np.log(1 - self.model.urd), np.log(1 + self.model.urd), n_samples
        )

        mean = np.mean(log1pdelta)
        var = np.var(log1pdelta)
        bound = np.max(np.abs(log1pdelta))

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _plot_pdf(self, n_samples: int = 100000):
        # log(1 + delta)
        log1pdelta = np.random.uniform(
            np.log(1 - self.model.urd), np.log(1 + self.model.urd), n_samples
        )
        # delta
        delta = np.exp(log1pdelta) - 1

        fig, axs = plt.subplots(
            1, 2, figsize=(10, 3), sharey=True, sharex=True, layout="compressed"
        )
        axs[0].hist(delta, density=True, bins=50)
        axs[0].set_xlabel(r"$\delta$")
        axs[0].set_ylabel(r"$f_{\delta}(\delta)$")
        axs[1].hist(log1pdelta, density=True, bins=50)
        axs[1].set_xlabel(r"$\log(1 + \delta)$")
        axs[1].set_ylabel(r"$f_{\log(1+\delta)}(\log(1+\delta)$")
        plt.savefig("uniform_log1pdelta_model_pdf.png")
        plt.close()

    def _plot_cdf(self, n_samples: int = 100000):
        # log(1 + delta)
        log1pdelta = np.random.uniform(
            np.log(1 - self.model.urd), np.log(1 + self.model.urd), n_samples
        )
        cdf_func_log1pdelta, log1pdelta_sorted, cdf_log1pdelta = empirical_cdf(
            log1pdelta
        )
        # delta
        delta = np.exp(log1pdelta) - 1
        cdf_func_delta, delta_sorted, cdf_delta = empirical_cdf(delta)

        fig, axs = plt.subplots(
            1, 2, figsize=(10, 4), sharey=True, sharex=True, layout="compressed"
        )
        axs[0].plot(delta_sorted, cdf_delta)
        axs[0].set_xlabel(r"$\delta$")
        axs[0].set_ylabel(r"$F_{\delta}(\delta)$")
        axs[1].plot(log1pdelta_sorted, cdf_log1pdelta)
        axs[1].set_xlabel(r"$\log(1 + \delta)$")
        axs[1].set_ylabel(r"$F_{\log(1+\delta)}(\log(1+\delta)$")
        plt.savefig("uniform_log1pdelta_model_cdf.png")
        plt.close()


class beta_log_model:
    """
    Y is a beta distribution bounded by log(1-u) and log(1+u)
    Y ~ log(1-u) + (log(1+u) - log(1-u)) * Z;
    """

    def __init__(self, model="single", alpha: float = 3.0, beta: float = 2.0):
        self.alpha = alpha
        self.beta = beta
        self.model = IEEEModel(model)
        self._plot_pdf(n_samples=50000)  # probability density function
        self._plot_cdf(n_samples=50000)  # cummulative distribution function

        # stats

        # self.delta_stats = self._get_delta_statistics()
        self.delta_emperical_stats = self._get_delta_emperical_statistics()

        self.log1pdelta_stats = self._get_log1pdelta_statistics()
        self.log1pdelta_emperical_stats = self._get_log1pdelta_emperical_statistics()

        print("==" * 10 + " Log Beta Model " + "==" * 10)
        self._evaluate_mean_condition()
        print(
            "[Delta emperical stats] :"
            + ", ".join(f"{k}: {v:.3e}" for k, v in self.delta_emperical_stats.items())
        )
        print(
            "[Log(1+Delta) stats] :"
            + ", ".join(f"{k}: {v:.3e}" for k, v in self.log1pdelta_stats.items())
        )
        print(
            "[Log(1+Delta) emperical stats] :"
            + ", ".join(
                f"{k}: {v:.3e}" for k, v in self.log1pdelta_emperical_stats.items()
            )
        )

    def _evaluate_mean_condition(self):
        """evaluate the sign of mean of delta"""
        u = self.model.urd
        L = np.log((1 + u) / (1 - u))
        c = -np.log(1 - u) / L
        p = self.alpha / (self.alpha + self.beta)
        if p > c:
            print("Mean of delta is strictly positive")
        elif p == c:
            print("Mean of delta is zero")
        elif p < c:
            print("Mean of delta is strictly negative")

    def _get_delta_statistics(self):
        """statistics of rounding error delta"""
        raise NotImplementedError("Closed form not evaluated yet.")

    def _get_log1pdelta_statistics(self):
        """statistics of log(1 + delta), with delta~U[-urd, urd]"""
        # unit roundoff
        u = self.model.urd
        # scale factor
        L = np.log((1 + u) / (1 - u))
        # mean
        mean = np.log(1 - u) + L * (self.alpha / (self.alpha + self.beta))
        # variance
        var = (
            L**2
            * self.alpha
            * self.beta
            / ((self.alpha + self.beta + 1) * (self.alpha + self.beta) ** 2)
        )
        # bound
        bound = np.log(1 + u)

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _get_delta_emperical_statistics(self, n_samples: int = 100000):
        # unit roundoff
        u = self.model.urd
        # sample from beta distribution B(alpha, beta)
        Z = np.random.beta(self.alpha, self.beta, size=n_samples)
        # log1pdelta
        L = np.log((1 + u) / (1 - u))
        log1pdelta = np.log(1 - u) + L * Z
        # delta
        delta = np.exp(log1pdelta) - 1

        mean = np.mean(delta)
        var = np.var(delta)
        bound = np.max(np.abs(delta))

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _get_log1pdelta_emperical_statistics(self, n_samples: int = 100000):
        # unit roundoff
        u = self.model.urd
        # sample from beta distribution B(alpha, beta)
        Z = np.random.beta(self.alpha, self.beta, size=n_samples)
        # log1pdelta
        L = np.log((1 + u) / (1 - u))
        log1pdelta = np.log(1 - u) + L * Z

        mean = np.mean(log1pdelta)
        var = np.var(log1pdelta)
        bound = np.max(np.abs(log1pdelta))

        stats = {
            "mean": mean,
            "var": var,
            "bound": bound,
        }

        return stats

    def _plot_pdf(self, n_samples: int = 100000):
        # unit roundoff
        u = self.model.urd
        # sample from beta distribution B(alpha, beta)
        Z = np.random.beta(self.alpha, self.beta, size=n_samples)
        # log1pdelta
        L = np.log((1 + u) / (1 - u))
        log1pdelta = np.log(1 - u) + L * Z
        # delta
        delta = np.exp(log1pdelta) - 1

        fig, axs = plt.subplots(
            1, 2, figsize=(10, 3), sharey=True, sharex=True, layout="compressed"
        )
        axs[0].hist(delta, density=True, bins=50)
        axs[0].set_xlabel(r"$\delta$")
        axs[0].set_ylabel(r"$f_{\delta}(\delta)$")
        axs[1].hist(log1pdelta, density=True, bins=50)
        axs[1].set_xlabel(r"$\log(1 + \delta)$")
        axs[1].set_ylabel(r"$f_{\log(1+\delta)}(\log(1+\delta)$")
        plt.savefig("beta_log_model_pdf.png")
        plt.close()

    def _plot_cdf(self, n_samples: int = 100000):
        # unit roundoff
        u = self.model.urd
        # sample from beta distribution B(alpha, beta)
        Z = np.random.beta(self.alpha, self.beta, size=n_samples)
        # log1pdelta
        L = np.log((1 + u) / (1 - u))
        log1pdelta = np.log(1 - u) + L * Z
        cdf_func_log1pdelta, log1pdelta_sorted, cdf_log1pdelta = empirical_cdf(
            log1pdelta
        )
        # delta
        delta = np.exp(log1pdelta) - 1
        cdf_func_delta, delta_sorted, cdf_delta = empirical_cdf(delta)

        fig, axs = plt.subplots(
            1, 2, figsize=(10, 4), sharey=True, sharex=True, layout="compressed"
        )
        axs[0].plot(delta_sorted, cdf_delta)
        axs[0].set_xlabel(r"$\delta$")
        axs[0].set_ylabel(r"$F_{\delta}(\delta)$")
        axs[1].plot(log1pdelta_sorted, cdf_log1pdelta)
        axs[1].set_xlabel(r"$\log(1 + \delta)$")
        axs[1].set_ylabel(r"$F_{\log(1+\delta)}(\log(1+\delta)$")
        plt.savefig("beta_log_model_cdf.png")
        plt.close()


if __name__ == "__main__":
    seed_everything()
    # uniform model (delta is uniform distribution)
    # uniform_model(model="half")
    # uniform log model (log(1+delta) is uniform distribution)
    # uniform_log_model(model="half")
    # # beta model (log(1 + delta) is beta distribution)
    beta_log_model(model="half", alpha=4.0, beta=2.0)
