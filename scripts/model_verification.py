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
        self.check_expected_delta_sign()

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
        # mean and var of Beta distribution
        mean_Z = self.alpha / (self.alpha + self.beta)
        var_Z = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1.0))

        # mean of the skewed distribution
        ell = np.log1p(urd) - np.log1p(-urd)
        mean_Y = np.log1p(-urd) +  ell * mean_Z
        var_Y = (ell)**2 * var_Z
        stats = {"mean": mean_Y, "var": var_Y, "bound": np.log1p(urd)}
        return stats

    def check_expected_delta_sign(self):
        urd = self.IEEE_model.urd
        ell = np.log1p(urd) - np.log1p(-urd)
        mean_Z = self.alpha / (self.alpha + self.beta)
        test = -np.log1p(-urd) / ell
        # negativitiy test
        if mean_Z < test:
            print("expected delta < 0 (strictly)")

        # positivity test
        log1p_stats = self.get_log1pstats()
        if log1p_stats["mean"] > 0.0:
            print("expected delta > 0 (strictly)")

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

    def get_log1pstats(self):
        urd = self.IEEE_model.urd
        mean = (-2.0*urd + (-1.0 + urd) * np.log1p(-urd) + (1.0 + urd) * np.log1p(urd)) / (2.0*urd)
        kappa = -1.0 + urd**2
        var = (4.0*urd**2 + kappa*(np.log1p(-urd)**2 - 2.0*np.log1p(-urd)*np.log1p(urd) + np.log1p(urd)**2)) / (4.0*urd**2)
        stats = {"mean": mean, "var": var, "bound": np.log1p(urd)}
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
        sn_samples = np.array(sn_samples)
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

class VPREA():
    def __init__(self, delta_model:BetaModel|UniformModel):
        self.delta_model = delta_model

    def get_one_minus_zeta(self, confidence, num_bounds_satisfied:int=1):
        Q = confidence
        m = num_bounds_satisfied

        return (1.0 - Q) / m

    def get_roots(self, n, one_minus_zeta):
        # get the log1p stats
        stats_log1p = self.delta_model.get_log1pstats()
        c = stats_log1p["bound"]
        sigma_sq = stats_log1p["var"]
        # quadratic coefficients
        a_quad = 1.0
        b_quad = (2.0/3.0)*c*(np.log(one_minus_zeta) - np.log(2.0))
        c_quad = 2.0 * n * sigma_sq * (np.log(one_minus_zeta) - np.log(2.0))

        t_plus = (-b_quad + np.sqrt(b_quad**2 - 4.0*a_quad*c_quad)) / (2.0*a_quad)
        t_minus = (-b_quad - np.sqrt(b_quad**2 - 4.0*a_quad*c_quad)) / (2.0*a_quad)

        return {"t_plus": t_plus, "t_minus": t_minus}

    def get_gamma(self, n, confidence:float):
        # compute one minus zeta
        one_minus_zeta = self.get_one_minus_zeta(confidence=0.99, num_bounds_satisfied=1)
        # compute the roots
        roots = self.get_roots(n, one_minus_zeta)
        # gamma
        mu = self.delta_model.get_log1pstats()["mean"]
        coeff = roots['t_plus'] + n * np.abs(mu)
        gamma = np.exp(coeff) - 1.0
        gamma[gamma>1.0] = 1.0
        return gamma


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
    dtype = "half"
    model = IEEEModel(dtype)
    true_dp = TrueDotModel(dtype)
    rng = np.random.default_rng()
    N = list(np.logspace(2, np.log10(n_max), n_res, dtype=int))
    # sample si
    # si = (rng.random(n_res).reshape(-1, 1) * N.reshape(-1, 1)).astype(T_CONVERT[dtype])
    si = true_dp.sample_sn_discrete(N, n_samples).astype(T_CONVERT[dtype]).reshape(n_res, n_samples)
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
    # axs[0].axvline(model.urd, color='r', linestyle='--', label=r'$\mathrm{u}$')
    # axs[0].axvline(-model.urd, color='r', linestyle='--', label='__nolegend__')
    axs[0].set_xlabel(r"$\delta$")
    axs[0].set_ylabel(r"$f_{\delta}(\delta)$")
    axs[0].grid(False, which='both')
    axs[0].minorticks_off()

    # axs[0].set_title(rf"$\mathbb{{E}}[\delta] = {np.mean(delta.ravel()):.2e}$")
    axs[1].scatter(N, np.mean(delta, axis=1), color='blue', s=10)
    axs[1].axhline(0.0, color='0.7', linestyle='-')
    axs[1].set_xlabel(r"$n$")
    axs[1].set_ylabel(r"$\mathbb{E}[\delta\vert S_{n}]$")
    axs[1].set_xscale("log")
    axs[1].set_xlim(100, n_max)
    threshold = 4e3
    mantissa, exponent = f"{threshold:.0e}".split("e")
    exponent = int(exponent)
    axs[1].axvspan(threshold, n_max, color='r', alpha=0.15, label=rf"$n \geq {mantissa}\times 10^{{{exponent}}}$")
    axs[1].set_title(r"$S_{n} = \sum_{i=1}^n X_i; \quad X_i ~\sim\mathcal{U}(0, 1)$")
    axs[1].legend()
    # fig.suptitle(f"Rounding Error Distribution for Adding Small Increments (n_max={n_max:.1e})", x=0.5, ha="center")
    plt.savefig("rounding_error_distribution_small_increments.png")

def compare_the_trajectory_of_delta(n_max:int=1000, n_samples:int=1000):
    """ compare the trajector of delta """
    dtype = "half"
    model = IEEEModel(dtype)

    true_dp = TrueDotModel(dtype)
    u_model = UniformModel(dtype)
    b_model = BetaModel(dtype, alpha=1.999, beta=2.0)

    mean_true_delta = np.zeros(n_max)
    mean_u_delta = np.zeros(n_max)
    mean_b_delta = np.zeros(n_max)

    # sample the delta trajectory
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="compressed")
    for ii in range(n_samples):
        # sample the true delta trajectory
        true_delta_traj = true_dp.sample_delta_trajectory(n=n_max, n_traj=1)
        # sample the uniform model trajectory
        u_delta_traj = u_model.sample_delta_trajectory(n=n_max, n_traj=1)
        # sample the beta model trajectory
        b_delta_traj = b_model.sample_delta_trajectory(n=n_max, n_traj=1)

        if ii == 0:
            mean_true_delta = true_delta_traj.ravel()
            mean_u_delta = u_delta_traj.ravel()
            mean_b_delta = b_delta_traj.ravel()
        else:
            mean_true_delta = mean_true_delta + (true_delta_traj.ravel() - mean_true_delta) / (ii + 1)
            mean_u_delta = mean_u_delta + (u_delta_traj.ravel() - mean_u_delta) / (ii + 1)
            mean_b_delta = mean_b_delta + (b_delta_traj.ravel() - mean_b_delta) / (ii + 1)

        axs[0].plot(true_delta_traj, color='k', alpha=0.03)
        axs[0].plot(u_delta_traj, color='b', alpha=0.03)
        axs[0].plot(b_delta_traj, color='goldenrod', alpha=0.03)

    axs[0].plot(mean_true_delta, color='k', label='True Dot Model')
    axs[0].plot(mean_u_delta, color='b', label='Uniform Model')
    axs[0].plot(mean_b_delta, color='goldenrod', label='Beta Model')
    axs[0].set_xlabel(r"$n$")
    axs[0].set_ylabel(r"$\mathbb{E}[\delta]$")
    axs[0].set_xscale("log")

    axs[1].hist(true_delta_traj.ravel(), bins=50, density=True, alpha=0.7, color='k', label=r'True $\delta$ distribution')
    axs[1].hist(u_delta_traj.ravel(), bins=50, density=True, alpha=0.7, color='blue', label=r'$\mathcal{U}$-model')
    axs[1].hist(b_delta_traj.ravel(), bins=50, density=True, alpha=0.7, color='goldenrod', label=r'$\beta$-model')
    axs[1].axvline(model.urd, color='r', linestyle='--', label=r'$\mathrm{u}$')
    axs[1].axvline(-model.urd, color='r', linestyle='--', label='__nolegend__')
    axs[1].set_xlabel(r"$\delta$")
    axs[1].set_ylabel(r"$f_{\delta}(\delta)$")
    axs[1].legend(loc="best")

    plt.savefig("trajectory_of_delta.png")

def compare_random_walk(n_max:int=1000, n_samples:int=1000):
    """ compare the random walk of delta """
    dtype = "half"
    model = IEEEModel(dtype)

    true_dp = TrueDotModel(dtype)
    u_model = UniformModel(dtype)
    b_model = BetaModel(dtype, alpha=1.5, beta=2.0)

    true_rv = np.zeros((n_max, n_samples))
    true_logrv = np.zeros((n_max, n_samples))
    u_rv = np.zeros((n_max, n_samples))
    u_logrv = np.zeros((n_max, n_samples))
    b_rv = np.zeros((n_max, n_samples))
    b_logrv = np.zeros((n_max, n_samples))

    # sample the delta trajectory
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), layout="compressed")
    alpha_samples = 0.3
    for ii in range(n_samples):

        # sample the true delta trajectory
        true_delta_traj = true_dp.sample_delta_trajectory(n=n_max, n_traj=1)
        true_rv[:, ii] = np.cumprod(1.0 + true_delta_traj, axis=0).ravel()
        true_logrv[:, ii] = np.cumsum(np.log1p(true_delta_traj), axis=0).ravel()

        # sample the uniform model trajectory
        u_delta_traj = u_model.sample_delta_trajectory(n=n_max, n_traj=1)
        u_rv[:, ii] = np.cumprod(1.0 + u_delta_traj, axis=0).ravel()
        u_logrv[:, ii] = np.cumsum(np.log1p(u_delta_traj), axis=0).ravel()

        # sample the beta model trajectory
        b_delta_traj = b_model.sample_delta_trajectory(n=n_max, n_traj=1)
        b_rv[:, ii] = np.cumprod(1.0 + b_delta_traj, axis=0).ravel()
        b_logrv[:, ii] = np.cumsum(np.log1p(b_delta_traj), axis=0).ravel()

    N = np.arange(1, n_max+1)
    n_std = 2.0
    mean_true_logrv, std_true_logrv = np.mean(true_logrv, axis=1), np.std(true_logrv, axis=1)
    mean_u_logrv, std_u_logrv = np.mean(u_logrv, axis=1), np.std(u_logrv, axis=1)
    mean_b_logrv, std_b_logrv = np.mean(b_logrv, axis=1), np.std(b_logrv, axis=1)

    axs.plot(N, mean_true_logrv, color='k', label='True')
    axs.fill_between(N, mean_true_logrv - n_std*std_true_logrv, mean_true_logrv + n_std*std_true_logrv, color='k', alpha=alpha_samples)

    axs.plot(N, mean_u_logrv, color='b', label=r'$\mathcal{U}$-model')
    axs.fill_between(N, mean_u_logrv - n_std*std_u_logrv, mean_u_logrv + n_std*std_u_logrv, color='b', alpha=alpha_samples)

    axs.plot(N, mean_b_logrv, color='goldenrod', label=r'$\beta$-model')
    axs.fill_between(N, mean_b_logrv - n_std*std_b_logrv, mean_b_logrv + n_std*std_b_logrv, color='goldenrod', alpha=alpha_samples)

    axs.set_ylabel(r"$\sum_{i=1}^n \log(1 + \delta_i)$")
    axs.set_xlabel(r"$n$")
    axs.set_xscale("log")
    axs.set_xlim(1, n_max)
    threshold = 4e3
    mantissa, exponent = f"{threshold:.0e}".split("e")
    exponent = int(exponent)
    axs.axvspan(threshold, n_max, color='r', alpha=0.15, label=rf"$n \geq {mantissa}\times 10^{{{exponent}}}$")

    axs.legend()
    plt.savefig("random_walk_of_product.png")

def compare_bound_growth():
    # vprea with uniform model
    dtype = "single"
    n = np.logspace(0, 8, 100, dtype=int)
    confidence = 0.99
    vprea_u = VPREA(UniformModel(dtype))

    fig, axs = plt.subplots(1, 1, figsize=(6, 4), layout="compressed")
    gamma_u = vprea_u.get_gamma(n=n, confidence = confidence)
    axs.plot(n, gamma_u, color='b', label=r'$\mathcal{U}$-model')
    # vprea with beta model
    alpha_range = [1.95, 1.97, 1.99]
    linestyles = ['--', '-.', ':']
    for ii, alpha in enumerate(alpha_range):
        print(f"Processing beta model with alpha={alpha}")
        b_model = BetaModel(dtype, alpha=alpha, beta=2.0)
        vprea_b = VPREA(b_model)
        gamma_b = vprea_b.get_gamma(n=n, confidence = confidence)

        axs.plot(n, gamma_b, color='goldenrod', label=rf'$\beta$-model $(\alpha={alpha}, \beta=2.0)$', linestyle=linestyles[ii])

    axs.set_xlabel(r"$n$")
    axs.set_ylabel(r"$\gamma$")
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlim(1, 1e8)
    axs.legend()
    plt.savefig("bound_growth.png")

if __name__ == "__main__":
    # seed
    seed_everything()
    # compare the rounding error distribution for addition and multiplication
    # compare_distribution_for_addition_and_multiplication()

    # compare the rounding error distribution when adding small number to large number
    # compare_distribution_for_small_increments(n_max=10000, n_samples=10000, n_res=200)

    # compare the trajectory of product
    # compare_the_trajectory_of_delta(n_max=10000, n_samples=100)

    # compare the random walk
    compare_random_walk(n_max=10000, n_samples=1000)

    # compare bound growth
    # compare_bound_growth()
