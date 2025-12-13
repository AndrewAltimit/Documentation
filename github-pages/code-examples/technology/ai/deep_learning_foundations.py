"""
Deep Learning Foundations

Implementation of fundamental deep learning concepts including universal approximation,
optimization landscape analysis, and neural tangent kernels.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalApproximation:
    """Demonstrations of universal approximation theorems"""

    @staticmethod
    def cybenko_theorem_demo(
        target_func: Callable, n_hidden: int, domain: Tuple[float, float]
    ) -> nn.Module:
        """
        Cybenko's theorem: A feedforward network with one hidden layer
        can approximate any continuous function on a compact set
        """

        class SingleHiddenLayer(nn.Module):
            def __init__(self, n_hidden):
                super().__init__()
                self.hidden = nn.Linear(1, n_hidden)
                self.output = nn.Linear(n_hidden, 1)

            def forward(self, x):
                x = torch.sigmoid(self.hidden(x))
                return self.output(x)

        # Generate training data
        x = torch.linspace(domain[0], domain[1], 1000).reshape(-1, 1)
        y = torch.tensor([target_func(xi.item()) for xi in x]).reshape(-1, 1)

        # Train network
        model = SingleHiddenLayer(n_hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(5000):
            pred = model(x)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    @staticmethod
    def depth_efficiency(
        input_dim: int, output_dim: int, target_complexity: str = "polynomial"
    ) -> Dict[str, int]:
        """
        Compare network sizes needed for different depths
        Deep networks can be exponentially more efficient
        """
        if target_complexity == "polynomial":
            # Polynomial of degree d requires O(d) depth or O(d^n) width
            shallow_params = input_dim**5  # Very wide shallow network
            deep_params = 5 * input_dim * 100  # Deep network with 5 layers

        elif target_complexity == "periodic":
            # Periodic functions benefit greatly from depth
            shallow_params = 2**input_dim  # Exponential in input dimension
            deep_params = input_dim * 100 * 10  # Linear in depth

        return {
            "shallow_network_params": shallow_params,
            "deep_network_params": deep_params,
            "efficiency_ratio": shallow_params / deep_params,
        }

    @staticmethod
    def barron_approximation_bound(target_func: Callable, n_neurons: int) -> float:
        """
        Barron's theorem: For functions with bounded Fourier transform,
        approximation error decreases as O(1/n) with n neurons
        """
        # Estimate Fourier transform bound (simplified)
        # In practice, compute ∫|ω||f̂(ω)|dω
        fourier_bound = 10.0  # Placeholder

        # Approximation error bound
        error_bound = fourier_bound / np.sqrt(n_neurons)
        return error_bound


class NeuralNetOptimization:
    """Analysis of optimization landscape in deep learning"""

    @staticmethod
    def loss_landscape_analysis(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        directions: List[torch.Tensor],
    ) -> np.ndarray:
        """
        Analyze loss landscape along specified directions
        Following Li et al. (2018) visualization method
        """
        # Save original parameters
        original_params = [p.clone() for p in model.parameters()]

        # Create grid
        alpha_range = np.linspace(-1, 1, 51)
        beta_range = np.linspace(-1, 1, 51)
        loss_surface = np.zeros((len(alpha_range), len(beta_range)))

        # Normalize directions
        dir1_norm = sum(torch.norm(d) for d in directions[0])
        dir2_norm = sum(torch.norm(d) for d in directions[1])

        for i, alpha in enumerate(alpha_range):
            for j, beta in enumerate(beta_range):
                # Move parameters along directions
                for p, p_orig, d1, d2 in zip(
                    model.parameters(), original_params, directions[0], directions[1]
                ):
                    p.data = p_orig + alpha * d1 / dir1_norm + beta * d2 / dir2_norm

                # Compute loss
                total_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in dataloader:
                        output = model(batch_x)
                        loss = F.cross_entropy(output, batch_y)
                        total_loss += loss.item()

                loss_surface[i, j] = total_loss / len(dataloader)

        # Restore original parameters
        for p, p_orig in zip(model.parameters(), original_params):
            p.data = p_orig

        return loss_surface

    @staticmethod
    def compute_hessian_eigenvalues(
        model: nn.Module,
        loss_fn: Callable,
        data: torch.Tensor,
        targets: torch.Tensor,
        top_k: int = 10,
    ) -> np.ndarray:
        """
        Compute top eigenvalues of loss Hessian
        Indicates sharpness/flatness of minima
        """
        # Compute loss
        output = model(data)
        loss = loss_fn(output, targets)

        # Compute Hessian using automatic differentiation
        params = list(model.parameters())
        n_params = sum(p.numel() for p in params)

        # For large networks, use Lanczos algorithm for top eigenvalues
        def hessian_vector_product(v):
            """Compute Hv without forming full Hessian"""
            grad = torch.autograd.grad(loss, params, create_graph=True)
            flat_grad = torch.cat([g.view(-1) for g in grad])

            grad_prod = torch.sum(flat_grad * v)
            hvp = torch.autograd.grad(grad_prod, params, retain_graph=True)

            return torch.cat([g.view(-1) for g in hvp])

        # Use power iteration for top eigenvalue
        v = torch.randn(n_params)
        v = v / torch.norm(v)

        eigenvalues = []
        for _ in range(top_k):
            for _ in range(100):  # Power iteration
                Hv = hessian_vector_product(v)
                eigenvalue = torch.dot(v, Hv)
                v = Hv / torch.norm(Hv)

            eigenvalues.append(eigenvalue.item())

            # Deflate for next eigenvalue
            # (simplified - proper implementation would use Lanczos)

        return np.array(eigenvalues)

    @staticmethod
    def gradient_noise_scale(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        n_samples: int = 100,
    ) -> float:
        """
        Compute gradient noise scale: tr(H @ Σ) / ||∇L||²
        Indicates batch size needed for convergence
        """
        # Compute full-batch gradient
        full_grad = None
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()

        full_grad = torch.cat([p.grad.view(-1).clone() for p in model.parameters()])
        full_grad_norm = torch.norm(full_grad)

        # Estimate gradient covariance
        grad_samples = []
        for _ in range(n_samples):
            # Sample single data point
            idx = np.random.randint(len(dataloader.dataset))
            x, y = dataloader.dataset[idx]
            x, y = x.unsqueeze(0), torch.tensor([y])

            # Compute gradient
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()

            grad = torch.cat([p.grad.view(-1).clone() for p in model.parameters()])
            grad_samples.append(grad)

        # Compute noise scale
        grad_samples = torch.stack(grad_samples)
        grad_cov = torch.cov(grad_samples.T)
        noise_trace = torch.trace(grad_cov)

        return (noise_trace / (full_grad_norm**2)).item()


class NeuralTangentKernel:
    """Neural Tangent Kernel theory for understanding deep learning"""

    @staticmethod
    def compute_ntk(
        model: nn.Module, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute empirical NTK: Θ(x,x') = <∇_θf(x), ∇_θf(x')>
        """
        # Get model outputs
        f1 = model(x1)
        f2 = model(x2)

        # Compute gradients
        params = list(model.parameters())

        grad1 = torch.autograd.grad(
            f1.sum(), params, retain_graph=True, create_graph=True
        )
        grad2 = torch.autograd.grad(
            f2.sum(), params, retain_graph=True, create_graph=True
        )

        # Compute inner product
        kernel = 0
        for g1, g2 in zip(grad1, grad2):
            kernel += torch.sum(g1.view(-1) * g2.view(-1))

        return kernel

    @staticmethod
    def infinite_width_prediction(
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        kernel_func: Callable,
        reg: float = 1e-6,
    ) -> torch.Tensor:
        """
        Prediction using infinite-width NTK
        f(x) = K(x, X_train) @ (K(X_train, X_train) + reg*I)^{-1} @ y_train
        """
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # Compute train kernel matrix
        K_train = torch.zeros((n_train, n_train))
        for i in range(n_train):
            for j in range(n_train):
                K_train[i, j] = kernel_func(X_train[i], X_train[j])

        # Compute test-train kernel matrix
        K_test_train = torch.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_test_train[i, j] = kernel_func(X_test[i], X_train[j])

        # Solve kernel regression
        K_train_reg = K_train + reg * torch.eye(n_train)
        weights = torch.linalg.solve(K_train_reg, y_train)

        return K_test_train @ weights

    @staticmethod
    def compute_cntk(depth: int, width: int, activation: str = "relu") -> Callable:
        """
        Compute Convolutional NTK for CNN architectures
        Returns kernel function for given architecture
        """

        def cntk_kernel(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            # Simplified CNTK computation
            # In practice, involves recursive computation through layers

            # Initial kernel (linear)
            K = torch.dot(x1.view(-1), x2.view(-1))

            # Apply nonlinearity through layers
            for _ in range(depth):
                if activation == "relu":
                    # ReLU: K_new = (K * arcsin(K/sqrt((K_xx * K_yy))) + sqrt((K_xx * K_yy - K²))) / π
                    K_xx = torch.dot(x1.view(-1), x1.view(-1))
                    K_yy = torch.dot(x2.view(-1), x2.view(-1))

                    normalized_K = K / torch.sqrt(K_xx * K_yy)
                    K = (
                        K * torch.arcsin(normalized_K) + torch.sqrt(K_xx * K_yy - K**2)
                    ) / np.pi

                # Scale by width
                K = K * width

            return K

        return cntk_kernel

    @staticmethod
    def finite_width_correction(model: nn.Module, n_samples: int = 1000) -> float:
        """
        Estimate finite-width corrections to NTK
        Measures deviation from infinite-width behavior
        """
        # Sample random inputs
        input_dim = list(model.parameters())[0].shape[1]
        X = torch.randn(n_samples, input_dim)

        # Compute empirical NTK at initialization
        ntk_init = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                k = NeuralTangentKernel.compute_ntk(
                    model, X[i].unsqueeze(0), X[j].unsqueeze(0)
                )
                ntk_init.append(k.item())

        # Re-initialize and compute again
        for p in model.parameters():
            p.data.normal_(0, 1 / np.sqrt(p.shape[0]))

        ntk_reinit = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                k = NeuralTangentKernel.compute_ntk(
                    model, X[i].unsqueeze(0), X[j].unsqueeze(0)
                )
                ntk_reinit.append(k.item())

        # Compute relative standard deviation
        ntk_init = np.array(ntk_init)
        ntk_reinit = np.array(ntk_reinit)

        rel_std = np.std(ntk_reinit - ntk_init) / np.mean(ntk_init)
        return rel_std
