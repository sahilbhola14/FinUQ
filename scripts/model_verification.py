"""
Comparison of phi = prod_{i=1}^n (1 + delta_i) where delta_i are
1. true error from adding small number to larger numbers (as in the case of dot products)
2. beta distribution model where Y = log(1-urd) + log((1+urd) / (1-urd)) * Z; Z~Beta(alpha, beta)
and delta = exp(Y) - 1
3. delta is a random variable with distribution U(-urd, urd).
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
plt.style.use("./journal.mplstyle")
T_CONVERT = { "double": np.float64, "single": np.float32, "half": np.float16 }
DEFAULT_SEED = 42

def seed_everything(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)

class IEEEModel:
    def __init__(self, dtype):
        self.dtype = dtype
        assert self.dtype in ["double", "single", "half"], "invalid dtype"
        self.precision = self._get_precision()
        self.base = 2
        self.exponent_range = self._get_exponent_range()
        self.urd = 0.5 * self.base ** (1.0 - self.precision)

    def _get_precision(self):
        precision = {"double": 53, "single": 24, "half": 11}
        return precision.get(self.dtype)

    def _get_exponent_range(self):
        exponent_range = {
            "double": [-1021, 1024],
            "single": [-125, 128],
            "half": [-13, 16],
        }
        return exponent_range.get(self.dtype)

class BetaModel:
    def __init__(self, dtype, alpha:float=2.2, beta:float=2.2):
        self.IEEE_model = IEEEModel(dtype)
        self.alpha = alpha
        self.beta = beta

    def sample_log1pdelta(self, n_samples):
        # sample from Beta distribution B(alpha, beta)
        Z = np.random.beta(self.alpha, self.beta, n_samples)
        # sample from the skewed distribution
        urd = self.IEEE_model.urd
        ell = np.log1p(urd) - np.log1p(-urd)
        Y = np.log1p(-urd) + ell * Z
        return Y

    def sample_delta(self, n_samples):
        return np.exp(self.sample_log1pdelta(n_samples)) - 1.0

    def sample_delta_trajectory(self, n:int, n_traj:int):
        delta = self.sample_delta(n * n_traj).reshape(n, n_traj)
        return delta

    def get_log1pstats(self):
        urd = self.IEEE_model.urd
        mean_Z = self.alpha / (self.alpha + self.beta)
        var_Z = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        mean_Y = np.log(1 - urd) + np.log((1 + urd) / (1 - urd)) * mean_Z
        var_Y = (np.log((1 + urd) / (1 - urd)))**2 * var_Z
        stats = {"mean": mean_Y, "var": var_Y}
        return stats

class TrueDotModel:
    def __init__(self, dtype):
        self.dtype = dtype
        self.IEEE_model = IEEEModel(dtype)
        self.convert = { "double": np.float64, "single": np.float32, "half": np.float16 }

    def sample_sn_discrete(self, n:list, n_samples:int=1):
        """ sample sn = sum_{i=1}^n ai where ai ~ U(0, 1) in the given dtype.
        """
        assert isinstance(n, list), "n must be a list of integers"
        """
        Sample S_n = sum_{i=1}^n a_i, a_i ~ U(0,1)
        Returns shape: (n, n_samples)
        """
        rng = np.random.default_rng()
        sn_samples = []
        for ni in n:
            # sample xi
            xi = rng.random(ni*n_samples).reshape(-1, n_samples)
            # sample si
            sn = np.sum(xi, axis=0)
            sn_samples.append(sn)
        sn_samples = np.array(sn_samples).astype(self.convert[self.dtype])
        return sn_samples

    def sample_sn_continuous(self, n:int, n_samples):
        rng = np.random.default_rng()
        x = rng.random(n * n_samples).reshape(n, n_samples)
        return np.cumsum(x, axis=0)

    def sample_delta_trajectory(self, n:int, n_traj:int):
        """
        1. sample sn
        """
        # possible n
        N = np.arange(1, n+1)
        # sample sn
        # si = self.sample_sn_continuous(N.tolist(), n_samples=n_traj).reshape(n, n_traj).astype(self.convert[self.dtype])
        si = self.sample_sn_continuous(n, n_samples=n_traj).astype(self.convert[self.dtype])
        # sample ai
        rng = np.random.default_rng()
        ai = rng.random(n * n_traj).reshape(n, n_traj).astype(self.convert[self.dtype])
        # computed sum
        computed_sum = si + ai
        # true sum
        true_sum = si.astype(np.float64) + ai.astype(np.float64)
        # delta trajectory
        delta_traj = (computed_sum.astype(np.float64) - true_sum) / true_sum

        assert delta_traj.shape == (n, n_traj), "invalid shape of delta_traj"
        return delta_traj


class UniformModel:
    def __init__(self, dtype):
        self.IEEE_model = IEEEModel(dtype)

    def sample_log1pdelta(self, n_samples):
        # sample delta
        delta = self.sample_delta(n_samples)
        # sample from the skewed distribution
        return np.log1p(delta)

    def sample_delta(self, n_samples):
        # sample from uniform distribution
        urd = self.IEEE_model.urd
        delta = -urd + 2.0*urd*np.random.rand(n_samples)
        return delta

    def sample_delta_trajectory(self, n:int, n_traj:int):
        delta = self.sample_delta(n * n_traj).reshape(n, n_traj)
        return delta

def compare_distribution_for_addition_and_multiplication():
    dtype = "single"
    model = IEEEModel(dtype)
    rng = np.random.default_rng()
    ai = rng.uniform(low=0.0, high=1.0, size=100000).astype(T_CONVERT[dtype])
    bi = rng.uniform(low=0.0, high=1.0, size=100000).astype(T_CONVERT[dtype])
    # computed sum
    computed_sum = ai + bi
    computed_product = ai * bi
    # true sum
    true_sum = (ai.astype(np.float64) + bi.astype(np.float64))
    true_product = (ai.astype(np.float64) * bi.astype(np.float64))
    # rounding error
    delta_sum = (computed_sum.astype(np.float64) - true_sum) / true_sum
    delta_product = (computed_product.astype(np.float64) - true_product) / true_product
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="compressed")
    axs[0].hist(delta_sum, bins=100, density=True, alpha=0.7, color='blue')
    axs[0].set_title("Addition Rounding Error Distribution", x=0.5, ha="center")
    axs[1].hist(delta_product, bins=100, density=True, alpha=0.7, color='blue')
    axs[1].set_title("Multiplication Rounding Error Distribution", x=0.5, ha="center")
    for ax in axs:
        ax.set_xlabel("Relative Rounding Error")
        ax.set_ylabel(r"$f_{\delta}(\delta)$")
    fig.align_labels()
    plt.savefig("rounding_error_distribution_addition_multiplication.png")

def compare_distribution_for_small_increments(n_max:int=1000, n_samples:int=10000, n_res:int=10):
    dtype = "single"
    model = IEEEModel(dtype)
    true_dp = TrueDotModel(dtype)
    rng = np.random.default_rng()
    N = list(np.logspace(2, np.log10(n_max), n_res, dtype=int))
    # sample si
    # si = (rng.random(n_res).reshape(-1, 1) * N.reshape(-1, 1)).astype(T_CONVERT[dtype])
    si = true_dp.sample_sn_discrete(N).astype(T_CONVERT[dtype]).reshape(n_res, 1)
    # sample the addition
    ai = rng.random(n_res * n_samples).reshape(n_res, n_samples).astype(T_CONVERT[dtype])
    # compute sum
    computed_sum = si + ai
    # true sum
    true_sum = si.astype(np.float64) + ai.astype(np.float64)
    # rounding error
    delta = (computed_sum.astype(np.float64) - true_sum) / true_sum

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="compressed")
    axs[0].hist(delta.ravel(), color='blue', bins=50, density=True, alpha=0.7)
    axs[0].set_xlabel(r"$\delta$")
    axs[0].set_ylabel(r"$f_{\delta}(\delta)$")
    # axs[0].set_title(rf"$\mathbb{{E}}[\delta] = {np.mean(delta.ravel()):.2e}$")
    axs[1].scatter(N, np.mean(delta, axis=1), color='blue', s=10)
    axs[1].axhline(0.0, color='0.7', linestyle='-')
    axs[1].set_xlabel(r"$n$")
    axs[1].set_ylabel(r"$\mathbb{E}[\delta\vert S_{i-1}=s_{i-1}]$")
    axs[1].set_xscale("log")
    axs[1].set_xlim(100, n_max)
    threshold = 3e7
    mantissa, exponent = f"{threshold:.0e}".split("e")
    exponent = int(exponent)
    axs[1].axvspan(threshold, n_max, color='r', alpha=0.15, label=rf"$n > {mantissa}\times 10^{{{exponent}}}$")
    axs[1].set_title(r"$S_{i-1} = \sum_{i=1}^n X_i; \quad X_i ~\sim{U}(0, 1)$")
    axs[1].legend()
    # fig.suptitle(f"Rounding Error Distribution for Adding Small Increments (n_max={n_max:.1e})", x=0.5, ha="center")
    plt.savefig("rounding_error_distribution_small_increments.png")

def compare_the_trajectory_of_delta(n_max:int=1000, n_samples:int=1000):
    """ compare the trajector of delta """
    dtype = "single"
    model = IEEEModel(dtype)
    true_dp = TrueDotModel(dtype)
    # sample the delta trajectory
    true_delta_traj = true_dp.sample_delta_trajectory(n=n_max, n_traj=n_samples)
    # sample the uniform model trajectory
    u_model = UniformModel(dtype)
    u_delta_traj = u_model.sample_delta_trajectory(n=n_max, n_traj=n_samples)
    # sample the beta model trajectory
    b_model = BetaModel(dtype, alpha=1.999, beta=2.0)
    b_delta_traj = b_model.sample_delta_trajectory(n=n_max, n_traj=n_samples)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout="compressed")
    axs[0].plot(true_delta_traj, color='k', alpha=0.3)
    axs[0].plot(u_delta_traj, color='b', alpha=0.3)
    axs[0].plot(b_delta_traj, color='goldenrod', alpha=0.3)
    axs[0].set_xlabel(r"$n$")
    axs[0].set_xscale("log")
    axs[1].hist(true_delta_traj.ravel(), bins=50, density=True, alpha=0.7, color='k', label='True Dot Model')
    axs[1].hist(u_delta_traj.ravel(), bins=50, density=True, alpha=0.7, color='blue', label='Uniform Model')
    axs[1].hist(b_delta_traj.ravel(), bins=50, density=True, alpha=0.7, color='goldenrod', label='Beta Model')
    axs[1].axvline(model.urd, color='r', linestyle='--', label=r'$\mathrm{u}$')
    axs[1].axvline(-model.urd, color='r', linestyle='--', label='__nolegend__')
    axs[1].set_xlabel(r"$\delta$")
    axs[1].set_ylabel(r"$f_{\delta}(\delta)$")
    axs[1].legend(loc="lower right")
    plt.savefig("trajectory_of_delta.png")

if __name__ == "__main__":
    # seed
    seed_everything()
    # experiment 1
    # compare the rounding error distribution for addition and multiplication
    # compare_distribution_for_addition_and_multiplication()
    # compare the rounding error distribution when adding small number to large number
    # compare_distribution_for_small_increments(n_max=100000000, n_samples=100000, n_res=500)
    # compare the trajectory of product
    compare_the_trajectory_of_delta(n_max=50000000, n_samples=1)


