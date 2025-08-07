"""
Code for Figure 1c of
Saul, L.K. and Jordan, M.I., 1997. A variational principle for model-based morphing.
In Advances in Neural Information Processing Systems (pp. 267-273).

Note not exactly the same setup and do not know settings of ell and variance to use
Also not sure how to deal with the change points

Written in a bit of a rush and not thoroughly tested so use with caution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as sla
from scipy.integrate import odeint
from scipy.optimize import fsolve
import time
import tqdm

NUM_DIM = 16

# TODO: if using different probabilities for clusters then shoudl adjust ell

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def func(x_concat, t, precision, ell, mean):
    # ODE defined in equation 9
    x = x_concat[:NUM_DIM][:, None] - mean[:, None]
    v = x_concat[NUM_DIM:][:, None]

    M = precision
    a_coeff = 0.5 * (x.T @ M @ x) + ell
    new_a = (x - v * (x.T @ M @ v)) / a_coeff

    derivs = np.vstack((v, new_a))
    return np.squeeze(derivs)


class Gaussian:
    def __init__(self, mean, precision):
        self.mean = mean
        self.precision = precision
        self._det_prec = sla.det(precision)
        self._ndim = mean.size
        # todo: pdf and log pdf does stuff very inefficiently if doing more than one. Move to using scipy stats.

    def pdf(self, x_points):
        if len(x_points.shape) == 1:
            x_points = x_points[:, None]
            mean = self.mean[:, None]

            first_factor = self._det_prec**0.5 / ((2 * np.pi) ** (self._ndim / 2.0))
            second_factor = np.exp(
                -0.5 * (x_points - mean).T @ self.precision @ (x_points - mean)
            )
        else:
            mean = self.mean[None, :]
            first_factor = self._det_prec**0.5 / ((2 * np.pi) ** (self._ndim / 2.0))
            second_factor = np.diagonal(
                np.exp(-0.5 * (x_points - mean) @ self.precision @ (x_points - mean).T)
            )[:, None]

        return first_factor * second_factor

    def log_pdf(self, x_points):
        if len(x_points.shape) == 1:
            x_points = x_points[:, None]
            mean = self.mean[:, None]
            third_comp = -0.5 * (x_points - mean).T @ self.precision @ (x_points - mean)
        else:
            mean = self.mean[None, :]
            third_comp = np.diagonal(
                -0.5 * (x_points - mean) @ self.precision @ (x_points - mean).T
            )[:, None]
        first_comp = 0.5 * np.log(self._det_prec)
        second_comp = -self._ndim / 2.0 * np.log(2 * np.pi)
        return first_comp + second_comp + third_comp


class PathOptimizer:
    def __init__(self, x_0, x_1, mog):
        self.x_0 = x_0
        self.x_1 = x_1
        self.mog = mog
        self.ell = 10

        self.q_t = _interp_2d(x_0, x_1, 20)
        self.z_t = self.solve_for_zt()

        self.max_iter = 50

    def solve_for_zt(self):
        """
        assigns the cluster associations for each time step based on Baye's rule
        """
        zt = []
        for i in range(self.q_t.shape[0]):
            x = self.q_t[i, :]
            ll = [g.log_pdf(x) for g in self.mog]
            cluster_assoc = np.argmax(ll)
            zt.append(cluster_assoc)
        return np.array(zt)

    def solve_for_qt(self):

        new_qts = []

        def recurse_next(new_qts, current_qts, current_zts, x_start, mog_id_start):
            if len(current_qts) == 0:
                return  # recursed down to bottom
            for t, zt in enumerate(current_zts):
                if zt != mog_id_start:
                    if t == 0:
                        # path was only one point so skip to next section
                        recurse_next(
                            new_qts,
                            current_qts[t + 1 :],
                            current_zts[t + 1 :],
                            x_start,
                            zt,
                        )
                        return
                    break
                else:
                    x_end = current_qts[t]

            path = self._solve_qt_around_one_gaussian(
                x_start, x_end, self.mog[mog_id_start]
            )

            # Next time in this function we move onto the next section
            next_qts = current_qts[t + 1 :]
            next_zts = current_zts[t + 1 :]

            # decided to not take whole path and actually stop one before. Find this converges better
            # and if do not do this then the change points stay fairly static. Not sure whether this
            # is the best way to deal with these change points, but it seems to give results similar
            #  to the paper.
            if len(next_qts) == 0:
                new_qts.append(path[:, :NUM_DIM])
            else:
                new_qts.append(path[:-1, :NUM_DIM])
                x_end = path[-2, :NUM_DIM]
            recurse_next(new_qts, next_qts, next_zts, x_end, zt)

        recurse_next(new_qts, self.q_t, self.z_t, self.x_0, self.z_t[0])

        new_qts_array = np.concatenate(new_qts, axis=0)
        return new_qts_array

    def _solve_qt_around_one_gaussian(self, x_start, x_end, gaussian):
        # we solve a path for one of the MOG factors by using shooting method
        precision = gaussian.precision
        mean = gaussian.mean
        tspan = np.linspace(0, 10.0, 50)

        def solutions_func(v_start_guess):
            U = odeint(
                func,
                np.concatenate((x_start, v_start_guess)),
                tspan,
                args=(precision, self.ell, mean),
            )
            u1 = U[:, :NUM_DIM]
            return u1[-1] - x_end

        vstart = fsolve(solutions_func, x_end)

        x0 = np.array(np.concatenate((x_start, vstart)), dtype=np.float32)
        sol = odeint(func, x0, tspan, args=(precision, self.ell, mean))
        return sol

    def iterate_and_solve_via_paper_approach(self):
        qts = [self.q_t]
        zts = [self.z_t]

        for i in range(self.max_iter):
            # First solve for new path route.
            new_qts = self.solve_for_qt()
            self.q_t = new_qts
            qts.append(self.q_t)

            # and then solve for which cluster each point on route belongs to.
            new_zts = self.solve_for_zt()
            self.z_t = new_zts
            zts.append(new_zts)
            print("Iteration {} finished".format(i))

        return qts, zts


def _create_isotropic_precision(var):
    return 1 / var * np.eye(NUM_DIM, dtype=float)


def _interp_2d(x_0, x_1, num):
    interps = np.array([np.linspace(a, b, num) for a, b in zip(x_0, x_1)]).T
    return interps


def plot(qts, zts, mog):
    X, Y = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(2, 8, 50))
    orig_shape = X.shape
    for j, (path_qt, path_zt) in enumerate(zip(qts, zts)):
        fig, ax = plt.subplots(figsize=(7, 9))

        for i, gaussian in enumerate(mog):

            z = gaussian.pdf(
                np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
            )
            Z = z.reshape(*orig_shape)
            c = colors[i]
            ax.contour(X, Y, Z, colors=c, alpha=0.5)
            ax.plot(path_qt[path_zt == i, 0], path_qt[path_zt == i, 1], "x", color=c)
            ax.plot(gaussian.mean[0], gaussian.mean[1], "*", color=c, ms=10)
        ax.plot(path_qt[0, 0], path_qt[0, 1], "o", color="k", ms=10)
        ax.plot(path_qt[-1, 0], path_qt[-1, 1], "o", color="k", ms=10)
        ax.set_title("Iteration {}".format(j))

        if j % 10 == 0 or j == 1 or j == 5:
            plt.savefig("pictures/iter_{}.png".format(j))


def main():
    mog = [
        Gaussian(np.array([-6.0, 3.5], dtype=float), _create_isotropic_precision(1.0)),
        Gaussian(np.array([-5.0, 6.2], dtype=float), _create_isotropic_precision(1.0)),
        Gaussian(np.array([-2.5, 8], dtype=float), _create_isotropic_precision(1.0)),
        Gaussian(np.array([2.5, 8], dtype=float), _create_isotropic_precision(1.0)),
        Gaussian(np.array([5.0, 6.2], dtype=float), _create_isotropic_precision(1.0)),
    ]
    x_0 = np.array([-6, 4.5], dtype=float)
    x_1 = np.array([6.0, 4.5], dtype=float)

    path_optimizer = PathOptimizer(x_0, x_1, mog)
    start_time = time.time()
    qts, zts = path_optimizer.iterate_and_solve_via_paper_approach()
    print(f"Optimization of 50 iterations took {time.time()-start_time:.2f} seconds.")
    plot(qts, zts, mog)


if __name__ == "__main__":
    main()
