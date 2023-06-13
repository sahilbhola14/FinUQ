# Script to examine finite precision errors while solving a stochastic PDE
# Author(s): Dr. Karthik Duraisamy, Sahil Bhola
# Date: 06/01/2023
import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
from mpi4py import MPI
import scipy.sparse as sp
import time
import scipy.io as sio

fig_type = 'present'  # present/paper


fig_quality = 500 if fig_type is 'present' else 10000

# Paper
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=20)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}' r'\usepackage{amsfonts}' r'\usepackage{amssymb}')
plt.rc('lines', linewidth=3, markersize=10, markeredgecolor='k', markeredgewidth=2)
plt.rc('axes', labelpad=20, grid=True)

np.random.seed(0)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class PDE():
    """PDE class
    Args:
    precision: precision of the computation

    Attributes:
    theta_sample: sample of the random variable
    num_intervals: number of intervals
    precision: precision of the variables
    spacing: spacing between the grid points
    grid: grid points
    diagonals: diagonals of the tridiagonal matrix
    rhs: right hand side of the PDE
    """
    def __init__(self, precision='double'):
        self.precision = precision
        assert (self.precision in ['half', 'single', 'double']), "Precision must be wither half, single or double"

    def _get_diagonals(self, theta_sample, spacing, num_intervals):
        """Function to get the diagonals of the tridiagonal matrix
        Note: Second order central difference scheme is used for all the derivatives
        Returns:
        diagonals: Diagonals of the tridiaognal matrix
        """
        half = self._change_precision(0.5)
        theta_1 = theta_sample[0]
        theta_2 = theta_sample[1]
        ld = (1 / spacing**2) * (1 + (theta_1*spacing*(np.arange(2, num_intervals) - half)))
        ud = (1 / spacing**2) * (1 + (theta_1*spacing*(np.arange(1, num_intervals-1) + half)))
        cd = (1 / spacing**2) * (-2 -2*theta_1*spacing*np.arange(1, num_intervals))

        ld = self._change_precision(ld)
        ud = self._change_precision(ud)
        cd = self._change_precision(cd)

        return (ld, cd, ud)

    def test_solution(self, solution, diagonals, rhs, num_intervals):
        """Function to test the solution
        """
        mat = sp.diags(diagonals, [-1, 0, 1], shape=(num_intervals-1, num_intervals-1)).toarray()
        rhs = rhs*np.ones(num_intervals-1)
        sol = np.linalg.solve(mat, rhs)

        assert np.linalg.norm(solution - sol) < 1e-6, "Solution error is too large"

    def _integrate(self, solution, spacing):
        """Fuction to compute the integral: \int_0^1 u(x) dx
        Args:
        solution: solution of the PDE
        Returns:
        integral: integral of the solution
        """
        return np.sum(solution)*spacing

    def comp_solution(self, theta_sample, num_intervals):
        """Function solves the PDE via Thomas algorithm
        Returns:
        solution: solution of the PDE
        """
        # Spacing
        spacing = 1 / num_intervals
        # Grid
        grid = np.linspace(0, 1, num_intervals + 1)
        # RHS
        rhs = -50*theta_sample[1]**2

        theta_sample = self._change_precision(theta_sample)
        spacing = self._change_precision(spacing)
        grid = self._change_precision(grid)
        diagonals = self._get_diagonals(theta_sample, spacing, num_intervals)
        rhs = self._change_precision(rhs)

        a = diagonals[1]  # main diagonal
        b = np.insert(diagonals[0], 0, np.nan)  # lower diagonal
        c = np.insert(diagonals[2], len(diagonals[2]), np.nan)  # upper diagonal

        solution = self._change_precision(np.zeros(num_intervals - 1))
        alpha = self._change_precision(np.zeros(num_intervals - 1))
        beta = self._change_precision(np.zeros(num_intervals - 1))

        alpha[0] = a[0]
        for ii in range(1, num_intervals-1):
            beta[ii] = b[ii]/alpha[ii-1]
            alpha[ii] = a[ii] - (beta[ii]*c[ii-1])

        y = self._change_precision(np.zeros(num_intervals - 1))
        y[0] = rhs
        for ii in range(1, num_intervals-1):
            y[ii] = rhs - (beta[ii]*y[ii-1])

        solution[-1] = y[-1]/alpha[-1]

        for ii in range(num_intervals-3, -1, -1):
            solution[ii] = y[ii]/alpha[ii] - (c[ii]*solution[ii+1])/alpha[ii]


        return self._integrate(solution, spacing)

    def _change_precision(self, input):
        """Function changes the precision
        Args:
        input: input to be changed to the desired precision
        Returns:
        output with the desired precision
        """
        if self.precision == 'half':
            return np.float16(input)
        elif self.precision == 'single':
            return np.float32(input)
        elif self.precision == 'double':
            return input


def analytical_solution(theta_sample):
    """Function returns the analytical solution of the PDE
    Args:
    theta_sample: sample of theta
    Returns:
    analytical_solution: analytical solution of the PDE
    """
    theta_1_sample = theta_sample[0]
    theta_2_sample = theta_sample[1]

    term1 = 25*(theta_2_sample**2)*(-2*theta_1_sample + (2 + theta_1_sample)*np.log(1 + theta_1_sample))
    term2 = (theta_1_sample**2)*np.log(1 + theta_1_sample)

    return term1/term2

def get_theta_sample(theta_dist):
    """Function returns the theta_sample
    Args:
    theta_dist: distribution of theta
    Returns:
    theta_sample: sample of theta
    """
    return theta_dist.sample()

def plot_solution_convergence(num_intervals, exact_solution, double_solution, single_solution, half_solution, num_samples):
    """Function plots the solution convergence
    Args:
    num_intervals: Invervals to plot
    exact_solution: Exact solution
    double_solution: Solution using double precision
    single_solution: Solution using single precision
    half_precision: Solution using single precision
    """
    if rank == 0:
        plt.figure()
        # plt.axhline(y=exact_solution, color='r', linestyle='-', label='Exact solution')

        # plt.plot( num_intervals, np.abs(exact_solution - double_solution), 'b', marker='D', label='Double precision')
        # plt.plot( num_intervals, np.abs(exact_solution - single_solution), '--m', marker='v', label='Single precision')
        # plt.plot( num_intervals, np.abs(exact_solution - half_solution), 'k', marker='^', label='Half precision')

        plt.plot(1 / num_intervals, np.abs(exact_solution - double_solution), 'b', marker='D', label='Double precision')
        plt.plot(1 / num_intervals, np.abs(exact_solution - single_solution), '--m', marker='v', label='Single precision')
        plt.plot(1 / num_intervals, np.abs(exact_solution - half_solution), 'k', marker='^', label='Half precision')

        plt.xlabel(r'$\Delta x$')
        plt.ylabel(r'$|p(\theta_1, \theta_2) - \hat{p}(\theta_1, \theta_2)|$')
        plt.legend()
        plt.grid()
        plt.savefig('solution_convergence_num_samples_{}.png'.format(num_samples), dpi=fig_quality, bbox_inches='tight')
        plt.close()


def main():
    # Begin User Input
    num_mcmc_samples = 10000
    num_intervals = 2**np.arange(2, 8)
    # End User Input
    # test_theta_1 = sio.loadmat('test_U.mat')['U'][:, 0]
    # test_theta_2 = sio.loadmat('test_Z.mat')['Z'][:, 0]
    # # plt.figure()
    # # plt.plot(test_theta_1, color="red")
    # # plt.plot(test_theta_2, color="blue")
    # # plt.show()
    # # breakpoint()

    exact_solution = 3.292903073382538  # Exact solution of the PDE
    theta_1_dist = cp.Uniform(0, 1) + 0.1
    theta_2_dist = cp.Normal(0, 1)
    theta_dist = cp.J(theta_1_dist, theta_2_dist)

    double_precision_pde = PDE(precision='double')
    single_precision_pde = PDE(precision='single')
    half_precision_pde = PDE(precision='half')


    num_mcmc_samples_per_proc = int(num_mcmc_samples / size)

    double_precision_solution = np.zeros((num_mcmc_samples_per_proc, len(num_intervals))).astype('float64')
    single_precision_solution = np.zeros((num_mcmc_samples_per_proc, len(num_intervals))).astype('float32')
    half_precision_solution = np.zeros((num_mcmc_samples_per_proc, len(num_intervals))).astype('float16')

    double_precision_pde.comp_solution(np.array([0.5, 0.5]), num_intervals=4)
    # theta_samples = np.array([[0.2465, 0.8395],[-0.7655, 0.5610]])

    tic = time.time()
    for ii in range(num_mcmc_samples_per_proc):
        print(ii)
        theta_sample = get_theta_sample(theta_dist)
        # theta_sample = np.array([test_theta_1[ii], test_theta_2[ii]]).reshape(-1, 1)
        # theta_sample = theta_samples[:, ii].reshape(-1, 1)
        for jj, interval in enumerate(num_intervals):
            # if rank == 0:
            #     print(ii, jj)
            double_precision_solution[ii, jj] = double_precision_pde.comp_solution(theta_sample, interval)
            single_precision_solution[ii, jj] = single_precision_pde.comp_solution(theta_sample, interval)
            half_precision_solution[ii, jj] = half_precision_pde.comp_solution(theta_sample, interval)


    # fig, axs = plt.subplots(figsize=(10, 5))
    # axs.axhline(y=exact_solution, color='k', linestyle='-', label='Exact solution')
    # axs.plot(1 / num_intervals, np.mean(double_precision_solution, axis=0), marker="D", color="blue", label='Double precision')
    # axs.plot(1 / num_intervals, np.mean(single_precision_solution, axis=0), '--', marker="v", color="green", label='Single precision')
    # axs.plot(1 / num_intervals, np.mean(half_precision_solution, axis=0), marker="^", color="red", label='Half precision')
    # axs.legend(loc="upper right")
    # axs.set_xlabel(r'$\Delta x$')
    # axs.set_ylabel(r'$p(\theta_1, \theta_2)$')
    # # axs.grid(visible=True, which="both")
    # plt.tight_layout()
    # plt.savefig('solution_convergence_matlab_samples.png', dpi=fig_quality, bbox_inches='tight')
    # plt.close()

    fig, axs = plt.subplots(figsize=(10, 5))
    axs.plot(1 / num_intervals, exact_solution - np.mean(double_precision_solution, axis=0), marker="D", color="blue", label='Double precision')
    axs.plot(1 / num_intervals, exact_solution - np.mean(single_precision_solution, axis=0), '--', marker="v", color="green", label='Single precision')
    axs.plot(1 / num_intervals, exact_solution - np.mean(half_precision_solution, axis=0), marker="^", color="red", label='Half precision')
    axs.legend(loc="upper right")
    axs.set_xlabel(r'$\Delta x$')
    axs.set_ylabel(r'$p(\theta_1, \theta_2) - \frac{1}{N}\sum_{i=1}^{N}\hat{p}(\theta_1, \theta_2)$')
    # axs.grid(visible=True, which="both")
    axs.set_ylim([-0.2, 0.2])
    plt.tight_layout()
    plt.savefig('error_num_mcmc_samples_{}.png'.format(num_mcmc_samples), dpi=fig_quality, bbox_inches='tight')
    plt.close()

    # sum_double_precision_solution = np.sum(double_precision_solution, axis=0)
    # sum_single_precision_solution = np.sum(single_precision_solution, axis=0)
    # sum_half_precision_solution = np.sum(half_precision_solution, axis=0)

    # double_precision_solution_global = comm.allreduce(sum_double_precision_solution, op=MPI.SUM)
    # double_precision_solution_global = double_precision_solution_global / num_mcmc_samples

    # single_precision_solution_global = comm.allreduce(sum_single_precision_solution, op=MPI.SUM)
    # single_precision_solution_global = single_precision_solution_global / num_mcmc_samples

    # half_precision_solution_global = comm.allreduce(sum_half_precision_solution, op=MPI.SUM)
    # half_precision_solution_global = half_precision_solution_global / num_mcmc_samples

    # # if rank == 0:
    # #     print(double_precision_solution_global, single_precision_solution_global, half_precision_solution_global)
    # #     print(double_precision_solution_global.dtype, single_precision_solution_global.dtype, half_precision_solution_global.dtype)

    # toc = time.time()

    # if rank == 0:
    #     print("Time taken for computation: ", toc - tic)

    # plot_solution_convergence(num_intervals=num_intervals,
    #                           exact_solution=exact_solution,
    #                           double_solution=double_precision_solution_global,
    #                           single_solution=single_precision_solution_global,
    #                           half_solution=half_precision_solution_global,
    #                           num_samples=num_mcmc_samples)



if __name__ == '__main__':
    main()
