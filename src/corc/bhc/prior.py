# -*- coding: utf-8 -*-

# License: GPL 3.0

import numpy as np
from numpy.linalg import linalg
from numpy.linalg import slogdet
from scipy.special.spfun_stats import multigammaln

from corc.bhc.api import AbstractPrior

LOG2PI = np.log(2 * np.pi)
LOG2 = np.log(2)


class NormalInverseWishart(AbstractPrior):
    """
    Reference: MURPHY, Kevin P.
               Conjugate Bayesian analysis of the Gaussian distribution.
               def, v. 1, n. 2σ2, p. 16, 2007.
               https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/readings/bayesGauss.pdf
    """

    def __init__(self, s_mat, r, v, m):
        self.s_mat = s_mat
        self.r = r
        self.v = v
        self.m = m
        self.log_prior0 = NormalInverseWishart.__calc_log_prior(s_mat, r, v)

    def calc_log_mlh(self, x_mat):
        x_mat_l = x_mat.copy()
        x_mat_l = x_mat_l[np.newaxis] if x_mat_l.ndim == 1 else x_mat_l
        n, d = x_mat_l.shape
        s_mat_p, rp, vp = NormalInverseWishart.__calc_posterior(
            x_mat_l, self.s_mat, self.r, self.v, self.m
        )
        log_prior = NormalInverseWishart.__calc_log_prior(s_mat_p, rp, vp)
        return log_prior - self.log_prior0 - LOG2PI * (n * d / 2.0)

    def calc_log_mlh_two(self, x1, x2):
        """
        Log-marginal likelihood of a NIW prior given *exactly two*
        data vectors ``x1`` and ``x2`` (both 1-D ``(d,)`` arrays).
        """
        # ---- constants that depend only on the prior -----------------
        d = x1.shape[0]  # dimensionality
        n = 2  # number of points

        # ---- sufficient statistics for the two points -------------
        x_bar = 0.5 * (x1 + x2)  # sample mean
        diff = x1 - x2
        S = 0.5 * np.outer(diff, diff)  # scatter matrix for n=2

        # ---- posterior hyper‑parameters ----------------------------
        rp = self.r + n  # r′
        vp = self.v + n  # v′
        dt = (x_bar - self.m)[np.newaxis]  # (1, d) column vector
        s_mat_p = self.s_mat + S + (self.r * n / rp) * np.dot(dt.T, dt)

        # ---- log‑posterior and final log‑likelihood ----------------
        log_post = NormalInverseWishart.__calc_log_prior(s_mat_p, rp, vp)
        # the “data” term is  -log(2π) * n*d/2  (same as in calc_log_mlh)
        return log_post - self.log_prior0 - LOG2PI * (n * d / 2.0)

    def pairwise_niw_log_mlh_numpy(
        self,
        X,  # (N, d) data matrix
    ):
        """
        Returns an (N, N) array L where L[i, j] = log p({X[i], X[j]} | NIW).
        Diagonal entries are set to ``np.nan`` (the “pair” of a point with itself is not defined).

        The formula is the NIW marginal likelihood for exactly two observations:
            log p = log_prior_post - log_prior_0 - LOG2PI * d   (because n=2)
        where
            log_prior_0 = __calc_log_prior(s_mat, r, v)
            log_prior_post = __calc_log_prior(s_mat_p, r+2, v+2)
        and
            s_mat_p = s_mat + S + (r·2/(r+2))·(x̄‑m)(x̄‑m)ᵀ
            S      = ½ (x_i‑x_j)(x_i‑x_j)ᵀ
            x̄      = ½ (x_i + x_j)
        """
        s_mat = self.s_mat
        r = self.r
        v = self.v
        m = self.m

        N, d = X.shape
        if d != s_mat.shape[0]:
            raise ValueError("data dimension and prior scale matrix do not match")

        # ------------------------------------------------------------------
        # 1. Constant part that depends only on the prior (log_prior_0)
        # ------------------------------------------------------------------
        log_prior0 = (
            np.log(2) * (v * d / 2.0)
            + (d / 2.0) * np.log(2.0 * np.pi / r)
            + multigammaln(v / 2.0, d)
            - (v / 2.0) * np.log(np.linalg.det(s_mat))
        )

        # ------------------------------------------------------------------
        # 2. Pairwise sufficient statistics (broadcasted)
        # ------------------------------------------------------------------
        # (N, N, d) arrays:   diff = x_i - x_j,  x_bar = (x_i + x_j)/2
        diff = X[:, None, :] - X[None, :, :]  # shape (N, N, d)
        x_bar = 0.5 * (X[:, None, :] + X[None, :, :])  # shape (N, N, d)

        # Scatter matrix for a pair:  S = ½ diff·diffᵀ   → (N, N, d, d)
        # Using einsum avoids an intermediate (N,N,d,1) reshape
        S = 0.5 * np.einsum("...i,...j->...ij", diff, diff)  # (N,N,d,d)

        # Term (r·2/(r+2))·(x̄‑m)(x̄‑m)ᵀ
        dt = x_bar - m  # (N,N,d)
        outer_dt = np.einsum("...i,...j->...ij", dt, dt)  # (N,N,d,d)
        term = (r * 2.0 / (r + 2.0)) * outer_dt

        # Posterior scale matrix for each pair
        # Broadcast s_mat to (N,N,d,d) and add the two pair‑wise contributions
        s_mat_p = s_mat[None, None, :, :] + S + term  # (N,N,d,d)

        # ------------------------------------------------------------------
        # 3. Log‑posterior for each pair:   __calc_log_prior(s_mat_p, r+2, v+2)
        # ------------------------------------------------------------------
        rp = r + 2.0
        vp = v + 2.0

        # slogdet works on the last two axes, so we can call it directly on s_mat_p
        sign, logdet = slogdet(s_mat_p)  # both have shape (N, N)
        if not np.all(sign > 0):
            raise ValueError(
                "Posterior scale matrix not positive‑definite for some pair"
            )

        log_prior_post = (
            np.log(2) * (vp * d / 2.0)
            + (d / 2.0) * np.log(2.0 * np.pi / rp)
            + multigammaln(vp / 2.0, d)
            - (vp / 2.0) * logdet
        )  # (N,N)

        # ------------------------------------------------------------------
        # 4. Final log‑likelihood for each unordered pair
        # ------------------------------------------------------------------
        # The data‑term is the same for every pair because n=2:
        data_term = LOG2PI * d  # = LOG2PI * (n*d/2) with n=2

        L = log_prior_post - log_prior0 - data_term  # (N,N)

        # Diagonal entries are not defined (a point paired with itself)
        np.fill_diagonal(L, np.nan)

        return L

    def row_of_log_likelihood_for_pairs(
        self,
        X,  # (N, d) data matrix
        i,  # index of the row you want (int)
    ):
        """
        Returns a 1‑D ``np.ndarray`` containing the log‑likelihoods for the
        pairs (i, j) with j > i only.  Length of the array is ``N‑i‑1``.
        The element for ``j = i`` is omitted (pair with itself is undefined).
        """
        s_mat = self.s_mat
        r = self.r
        v = self.v
        m = self.m
        N, d = X.shape
        if d != s_mat.shape[0]:
            raise ValueError("data dimension and prior scale matrix do not match")

        # ------------------------------------------------------------------
        # 1. Constant part that depends only on the prior (log_prior_0)
        # ------------------------------------------------------------------
        log_prior0 = (
            np.log(2) * (v * d / 2.0)
            + (d / 2.0) * np.log(2.0 * np.pi / r)
            + multigammaln(v / 2.0, d)
            - (v / 2.0) * np.log(np.linalg.det(s_mat))
        )

        # ------------------------------------------------------------------
        # 2. Pairwise sufficient statistics – only for j > i
        # ------------------------------------------------------------------
        # slice of points that matter
        Xj = X[i + 1 :]  # shape (N-i-1, d)
        diff = X[i] - Xj  # broadcasted automatically
        # x_bar[j] = (X[i] + Xj[j]) / 2
        x_bar = 0.5 * (X[i] + Xj)  # (N-i-1, d)

        # Scatter matrix S = ½ diff·diffᵀ  → (N-i-1, d, d)
        S = 0.5 * np.einsum("...i,...j->...ij", diff, diff)

        # Term (r·2/(r+2))·(x̄‑m)(x̄‑m)ᵀ
        dt = x_bar - m  # (N-i-1, d)
        outer_dt = np.einsum("...i,...j->...ij", dt, dt)  # (N-i-1, d, d)
        term = (r * 2.0 / (r + 2.0)) * outer_dt

        # Posterior scale matrix for each pair
        s_mat_p = s_mat[None, :, :] + S + term  # (N-i-1, d, d)

        # ------------------------------------------------------------------
        # 3. Log‑posterior for each pair
        # ------------------------------------------------------------------
        rp = r + 2.0
        vp = v + 2.0
        sign, logdet = slogdet(s_mat_p)  # (N-i-1,)

        sign, logdet = slogdet(s_mat_p)
        if not np.all(sign > 0):
            # add jitter and try again
            eps = 1e-6
            s_mat_p += eps * np.eye(s_mat_p.shape[-1])
            sign, logdet = slogdet(s_mat_p)
            if not np.all(sign > 0):
                raise ValueError(
                    "Posterior scale matrix not PD even after jitter; "
                    "check data or increase prior strength."
                )
        log_prior_post = (
            np.log(2) * (vp * d / 2.0)
            + (d / 2.0) * np.log(2.0 * np.pi / rp)
            + multigammaln(vp / 2.0, d)
            - (vp / 2.0) * logdet
        )  # (N-i-1,)

        # ------------------------------------------------------------------
        # 4. Final log‑likelihood for the required pairs
        # ------------------------------------------------------------------
        data_term = LOG2PI * d
        L_row = log_prior_post - log_prior0 - data_term  # (N-i-1,)

        return L_row

    @staticmethod
    def __calc_log_prior(s_mat, r, v):
        d = s_mat.shape[0]
        log_prior = LOG2 * (v * d / 2.0) + (d / 2.0) * np.log(2.0 * np.pi / r)
        determinant = linalg.det(s_mat)
        if determinant <= 0:
            determinant = linalg.det(s_mat + 1e-6 * np.eye(d))
        log_prior += multigammaln(v / 2.0, d) - (v / 2.0) * np.log(determinant)
        return log_prior

    @staticmethod
    def __calc_posterior(x_mat, s_mat, r, v, m):
        n = x_mat.shape[0]
        x_bar = np.mean(x_mat, axis=0)
        rp = r + n
        vp = v + n
        s_mat_t = np.zeros(s_mat.shape) if n == 1 else (n - 1) * np.cov(x_mat.T)
        dt = (x_bar - m)[np.newaxis]
        s_mat_p = s_mat + s_mat_t + (r * n / rp) * np.dot(dt.T, dt)
        return s_mat_p, rp, vp

    @staticmethod
    def create(data, g, scale_factor):
        degrees_of_freedom = data.shape[1] + 1
        data_mean = np.mean(data, axis=0)
        data_matrix_cov = np.cov(data.T)
        scatter_matrix = (data_matrix_cov / g).T

        return NormalInverseWishart(
            scatter_matrix, scale_factor, degrees_of_freedom, data_mean
        )
