"""
Advanced Machine Learning Algorithms

Implementation of advanced ML algorithms including Gaussian Processes,
Variational Inference, and other probabilistic methods.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import norm


class GaussianProcess:
    """Gaussian Process for regression with uncertainty quantification"""

    def __init__(self, kernel_func: Callable, noise_variance: float = 1e-6):
        self.kernel = kernel_func
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data"""
        self.X_train = X
        self.y_train = y

        n = X.shape[0]
        K = np.zeros((n, n))

        # Compute covariance matrix
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])

        # Add noise to diagonal
        K += self.noise_variance * np.eye(n)

        # Compute inverse for predictions
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at test points

        μ* = k*ᵀ K⁻¹ y
        σ*² = k** - k*ᵀ K⁻¹ k*
        """
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]

        # Compute k*
        k_star = np.zeros((n_train, n_test))
        for i in range(n_train):
            for j in range(n_test):
                k_star[i, j] = self.kernel(self.X_train[i], X_test[j])

        # Compute k**
        k_star_star = np.zeros(n_test)
        for i in range(n_test):
            k_star_star[i] = self.kernel(X_test[i], X_test[i])

        # Predictions
        mean = k_star.T @ self.K_inv @ self.y_train
        variance = k_star_star - np.diag(k_star.T @ self.K_inv @ k_star)

        return mean, np.sqrt(variance)

    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood for hyperparameter optimization"""
        n = self.X_train.shape[0]

        # log p(y|X) = -0.5 * (y^T K^{-1} y + log|K| + n log(2π))
        sign, logdet = np.linalg.slogdet(self.K_inv)

        return -0.5 * (
            self.y_train.T @ self.K_inv @ self.y_train + logdet + n * np.log(2 * np.pi)
        )

    def sample_posterior(self, X_test: np.ndarray, n_samples: int = 5) -> np.ndarray:
        """Sample functions from the posterior GP"""
        mean, std = self.predict(X_test)
        n_test = X_test.shape[0]

        # Sample from multivariate normal
        samples = np.zeros((n_samples, n_test))
        for i in range(n_samples):
            samples[i] = mean + std * np.random.normal(size=n_test)

        return samples


class VariationalInference:
    """Mean-field variational inference for Bayesian models"""

    def __init__(self, model_log_prob: Callable):
        self.model_log_prob = model_log_prob

    def elbo(
        self,
        params: Dict[str, np.ndarray],
        variational_params: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """
        Evidence Lower Bound (ELBO):
        ELBO = E_q[log p(x,z)] - E_q[log q(z)]
        """
        # Sample from variational distribution
        n_samples = 1000
        elbo_samples = []

        for _ in range(n_samples):
            # Sample z ~ q(z)
            z_sample = {}
            entropy = 0

            for name, (mean, log_std) in variational_params.items():
                std = np.exp(log_std)
                z_sample[name] = mean + std * np.random.normal(size=mean.shape)

                # Add entropy term
                entropy += 0.5 * np.sum(1 + 2 * log_std + np.log(2 * np.pi))

            # Compute log p(x,z)
            log_prob = self.model_log_prob(params, z_sample)

            elbo_samples.append(log_prob + entropy)

        return np.mean(elbo_samples)

    def optimize(
        self,
        params: Dict[str, np.ndarray],
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Optimize variational parameters using stochastic gradient ascent
        """
        # Initialize variational parameters
        var_params = {}
        for name, param in params.items():
            var_params[name] = (
                np.zeros_like(param),  # mean
                np.full_like(param, -1.0),  # log std
            )

        for iteration in range(n_iterations):
            # Compute gradients using automatic differentiation
            # (simplified - in practice use autograd/JAX)
            gradients = self._compute_elbo_gradients(params, var_params)

            # Update parameters
            for name in var_params:
                mean, log_std = var_params[name]
                grad_mean, grad_log_std = gradients[name]

                var_params[name] = (
                    mean + learning_rate * grad_mean,
                    log_std + learning_rate * grad_log_std,
                )

        return var_params

    def _compute_elbo_gradients(
        self,
        params: Dict[str, np.ndarray],
        var_params: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict:
        """Compute gradients of ELBO with respect to variational parameters"""
        # Simplified gradient computation
        # In practice, use automatic differentiation
        gradients = {}
        epsilon = 1e-5

        for name in var_params:
            mean, log_std = var_params[name]
            grad_mean = np.zeros_like(mean)
            grad_log_std = np.zeros_like(log_std)

            # Finite differences for mean
            for i in range(mean.size):
                var_params_plus = var_params.copy()
                var_params_plus[name] = (mean.copy(), log_std)
                var_params_plus[name][0].flat[i] += epsilon

                elbo_plus = self.elbo(params, var_params_plus)
                elbo_minus = self.elbo(params, var_params)

                grad_mean.flat[i] = (elbo_plus - elbo_minus) / epsilon

            # Finite differences for log_std
            for i in range(log_std.size):
                var_params_plus = var_params.copy()
                var_params_plus[name] = (mean, log_std.copy())
                var_params_plus[name][1].flat[i] += epsilon

                elbo_plus = self.elbo(params, var_params_plus)
                elbo_minus = self.elbo(params, var_params)

                grad_log_std.flat[i] = (elbo_plus - elbo_minus) / epsilon

            gradients[name] = (grad_mean, grad_log_std)

        return gradients


class BayesianOptimization:
    """Bayesian optimization using Gaussian Processes"""

    def __init__(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]],
        kernel_func: Callable,
        acquisition: str = "ei",
    ):
        self.objective = objective_func
        self.bounds = bounds
        self.kernel = kernel_func
        self.acquisition = acquisition
        self.gp = GaussianProcess(kernel_func)

        # History
        self.X_observed = []
        self.y_observed = []

    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function"""
        if len(self.X_observed) == 0:
            return np.ones(X.shape[0])

        mu, sigma = self.gp.predict(X)

        # Current best
        f_best = np.min(self.y_observed)

        # Calculate EI
        with np.errstate(divide="warn"):
            imp = f_best - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def upper_confidence_bound(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function"""
        if len(self.X_observed) == 0:
            return np.ones(X.shape[0])

        mu, sigma = self.gp.predict(X)
        return -mu + beta * sigma

    def probability_of_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Probability of Improvement acquisition function"""
        if len(self.X_observed) == 0:
            return np.ones(X.shape[0])

        mu, sigma = self.gp.predict(X)
        f_best = np.min(self.y_observed)

        with np.errstate(divide="warn"):
            Z = (f_best - mu - xi) / sigma
            poi = norm.cdf(Z)
            poi[sigma == 0.0] = 0.0

        return poi

    def optimize(
        self, n_iterations: int = 50, n_initial: int = 5
    ) -> Tuple[np.ndarray, float]:
        """Run Bayesian optimization"""

        # Initial random sampling
        for _ in range(n_initial):
            x = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            y = self.objective(x)
            self.X_observed.append(x)
            self.y_observed.append(y)

        # Convert to arrays and fit GP
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        self.gp.fit(X_obs, y_obs)

        # Optimization loop
        for _ in range(n_iterations - n_initial):
            # Generate candidate points
            X_candidates = self._generate_candidates(1000)

            # Compute acquisition function
            if self.acquisition == "ei":
                acquisition_values = self.expected_improvement(X_candidates)
            elif self.acquisition == "ucb":
                acquisition_values = self.upper_confidence_bound(X_candidates)
            elif self.acquisition == "poi":
                acquisition_values = self.probability_of_improvement(X_candidates)
            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition}")

            # Select best candidate
            best_idx = np.argmax(acquisition_values)
            x_next = X_candidates[best_idx]

            # Evaluate objective
            y_next = self.objective(x_next)

            # Update observations
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)

            # Refit GP
            X_obs = np.array(self.X_observed)
            y_obs = np.array(self.y_observed)
            self.gp.fit(X_obs, y_obs)

        # Return best found
        best_idx = np.argmin(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]

    def _generate_candidates(self, n_candidates: int) -> np.ndarray:
        """Generate random candidate points within bounds"""
        candidates = np.zeros((n_candidates, len(self.bounds)))
        for i, (low, high) in enumerate(self.bounds):
            candidates[:, i] = np.random.uniform(low, high, n_candidates)
        return candidates
