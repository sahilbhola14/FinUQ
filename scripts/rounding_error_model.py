# rounding error model
# author: Sahil Bhola, University of Michigan, 2025

import random
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("./journal.mplstyle")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


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
    delta ~ U[-urd, urd] and has a pdf 1.0 / (2 * urd)
    pdf of y = log(1 + delta) is exp(y) / (2 * urd);
    """

    def __init__(self, model="single"):
        self.model = IEEEModel(model)
        self._plot_pdf()  # probability density function
        self._plot_cdf()  # cummulative distribution function

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

    def _plot_pdf(self, n_samples: int = 100000):
        delta = np.random.uniform(-self.model.urd, self.model.urd, 1000)
        # delta_stats = self._get_delta_statistics()
        log1pdelta = np.log(1 + delta)
        # log1pdelta_stats = self._get_log1pdelta_statistics()
        fig, axs = plt.subplots(
            1, 2, figsize=(10, 3), sharey=True, sharex=True, layout="compressed"
        )
        axs[0].hist(delta, density=True, bins=50)
        axs[0].set_xlabel(r"$\delta$")
        axs[0].set_ylabel(r"Emperical density function")
        axs[1].hist(log1pdelta, density=True, bins=50)
        axs[1].set_xlabel(r"$\log(1 + \delta)$")
        plt.savefig("uniform_delta_model_density.png")
        plt.close()

    def _plot_cdf(self, n_samples: int = 100000):
        delta = np.random.uniform(-self.model.urd, self.model.urd, 1000)
        # delta_stats = self._get_delta_statistics()
        log1pdelta = np.log(1 + delta)
        # log1pdelta_stats = self._get_log1pdelta_statistics()
        fig, axs = plt.subplots(
            1, 2, figsize=(10, 3), sharey=True, sharex=True, layout="compressed"
        )
        axs[0].hist(delta, density=True, bins=50)
        axs[0].set_xlabel(r"$\delta$")
        axs[0].set_ylabel(r"Emperical density function")
        axs[1].hist(log1pdelta, density=True, bins=50)
        axs[1].set_xlabel(r"$\log(1 + \delta)$")
        plt.savefig("uniform_delta_model_density.png")
        plt.close()


if __name__ == "__main__":
    seed_everything()
    model = uniform_model(model="half")
