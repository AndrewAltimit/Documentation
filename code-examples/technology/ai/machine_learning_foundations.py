"""
Machine Learning Foundations

Mathematical foundations and theoretical frameworks for machine learning including
PAC learning, convex optimization, and kernel methods.
"""

from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class PACLearning:
    """Probably Approximately Correct (PAC) Learning Framework"""

    @staticmethod
    def vc_dimension_bound(vc_dim: int, m: int, delta: float) -> float:
        """
        Generalization bound using VC dimension

        With probability 1-δ, for all h in hypothesis class H:
        R(h) ≤ R̂(h) + √((vc_dim * (ln(2m/vc_dim) + 1) + ln(4/δ)) / m)

        Args:
            vc_dim: VC dimension of hypothesis class
            m: Number of training samples
            delta: Confidence parameter

        Returns:
            Generalization error bound
        """
        if m < vc_dim:
            return float("inf")

        complexity_term = vc_dim * (np.log(2 * m / vc_dim) + 1) + np.log(4 / delta)
        return np.sqrt(complexity_term / m)

    @staticmethod
    def rademacher_complexity(X: np.ndarray, hypothesis_class: List[Callable]) -> float:
        """
        Empirical Rademacher complexity

        R̂ₘ(H) = E_σ[sup_{h∈H} (1/m) Σᵢ σᵢh(xᵢ)]
        """
        m = X.shape[0]
        n_iterations = 1000

        complexities = []
        for _ in range(n_iterations):
            # Generate Rademacher random variables
            sigma = np.random.choice([-1, 1], size=m)

            # Compute supremum over hypothesis class
            sup_value = -np.inf
            for h in hypothesis_class:
                correlation = np.mean(sigma * h(X))
                sup_value = max(sup_value, correlation)

            complexities.append(sup_value)

        return np.mean(complexities)


class ConvexOptimization:
    """Convex optimization algorithms for ML"""

    @staticmethod
    def proximal_gradient_descent(
        f_grad: Callable,
        g_prox: Callable,
        x0: np.ndarray,
        step_size: float,
        n_iter: int,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Proximal gradient descent for composite optimization:
        minimize f(x) + g(x), where f is smooth and g is convex

        Algorithm:
        x_{k+1} = prox_{αg}(x_k - α∇f(x_k))
        """
        x = x0.copy()
        history = []

        for _ in range(n_iter):
            # Gradient step
            x_grad = x - step_size * f_grad(x)

            # Proximal step
            x = g_prox(x_grad, step_size)

            history.append(np.linalg.norm(x))

        return x, history

    @staticmethod
    def accelerated_gradient_descent(
        f_grad: Callable, x0: np.ndarray, L: float, n_iter: int
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Nesterov's accelerated gradient descent
        Achieves O(1/k²) convergence rate for smooth convex functions
        """
        x = x0.copy()
        y = x0.copy()
        t = 1.0
        history = []

        for k in range(n_iter):
            x_new = y - (1 / L) * f_grad(y)
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x_new + ((t - 1) / t_new) * (x_new - x)

            x = x_new
            t = t_new

            history.append(np.linalg.norm(f_grad(x)))

        return x, history

    @staticmethod
    def admm(
        A: np.ndarray, b: np.ndarray, lambda_reg: float, rho: float, n_iter: int
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Alternating Direction Method of Multipliers (ADMM) for Lasso

        minimize (1/2)||Ax - b||² + λ||z||₁
        subject to x = z
        """
        n = A.shape[1]
        x = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)

        # Pre-compute matrix inverse
        ATA = A.T @ A
        ATb = A.T @ b
        factor = np.linalg.inv(ATA + rho * np.eye(n))

        history = []

        for _ in range(n_iter):
            # x-update
            x = factor @ (ATb + rho * (z - u))

            # z-update (soft thresholding)
            z_old = z
            z = np.sign(x + u) * np.maximum(np.abs(x + u) - lambda_reg / rho, 0)

            # u-update
            u = u + x - z

            # Track primal residual
            primal_residual = np.linalg.norm(x - z)
            history.append(primal_residual)

        return x, z, history


class KernelTheory:
    """Theoretical foundations of kernel methods"""

    @staticmethod
    def mercer_kernel_expansion(
        kernel_func: Callable, eigenvalues: np.ndarray, eigenfunctions: List[Callable]
    ) -> Callable:
        """
        Mercer's theorem: K(x,y) = Σᵢ λᵢ φᵢ(x) φᵢ(y)
        """

        def expanded_kernel(x, y):
            result = 0
            for i, (λ, φ) in enumerate(zip(eigenvalues, eigenfunctions)):
                result += λ * φ(x) * φ(y)
            return result

        return expanded_kernel

    @staticmethod
    def rkhs_norm(
        f: Callable,
        kernel_func: Callable,
        support_points: np.ndarray,
        coefficients: np.ndarray,
    ) -> float:
        """
        Compute RKHS norm: ||f||²_K = Σᵢⱼ αᵢ αⱼ K(xᵢ, xⱼ)
        """
        n = len(support_points)
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                K[i, j] = kernel_func(support_points[i], support_points[j])

        return np.sqrt(coefficients.T @ K @ coefficients)

    @staticmethod
    def kernel_ridge_regression(
        X: np.ndarray, y: np.ndarray, kernel_func: Callable, lambda_reg: float
    ) -> Callable:
        """
        Kernel ridge regression with representer theorem
        f*(x) = Σᵢ αᵢ K(xᵢ, x), where α = (K + λI)⁻¹y
        """
        n = X.shape[0]
        K = np.zeros((n, n))

        # Compute Gram matrix
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel_func(X[i], X[j])

        # Solve for coefficients
        alpha = np.linalg.solve(K + lambda_reg * np.eye(n), y)

        # Return prediction function
        def predict(x_new):
            return sum(alpha[i] * kernel_func(X[i], x_new) for i in range(n))

        return predict

    @staticmethod
    def kernel_pca(
        X: np.ndarray, kernel_func: Callable, n_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kernel Principal Component Analysis

        Returns:
            Eigenvectors and eigenvalues of centered kernel matrix
        """
        n = X.shape[0]
        K = np.zeros((n, n))

        # Compute kernel matrix
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel_func(X[i], X[j])

        # Center kernel matrix
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Normalize eigenvectors
        for i in range(n_components):
            eigenvectors[:, i] /= np.sqrt(eigenvalues[i])

        return eigenvectors[:, :n_components], eigenvalues[:n_components]

    @staticmethod
    def maximum_mean_discrepancy(
        X: np.ndarray, Y: np.ndarray, kernel_func: Callable
    ) -> float:
        """
        Maximum Mean Discrepancy (MMD) for two-sample testing

        MMD²(P, Q) = E[k(X, X')] - 2E[k(X, Y)] + E[k(Y, Y')]
        """
        n_x = X.shape[0]
        n_y = Y.shape[0]

        # E[k(X, X')]
        xx_sum = 0
        for i in range(n_x):
            for j in range(i + 1, n_x):
                xx_sum += kernel_func(X[i], X[j])
        xx_term = 2 * xx_sum / (n_x * (n_x - 1))

        # E[k(X, Y)]
        xy_sum = 0
        for i in range(n_x):
            for j in range(n_y):
                xy_sum += kernel_func(X[i], Y[j])
        xy_term = xy_sum / (n_x * n_y)

        # E[k(Y, Y')]
        yy_sum = 0
        for i in range(n_y):
            for j in range(i + 1, n_y):
                yy_sum += kernel_func(Y[i], Y[j])
        yy_term = 2 * yy_sum / (n_y * (n_y - 1))

        mmd_squared = xx_term - 2 * xy_term + yy_term
        return np.sqrt(max(0, mmd_squared))
