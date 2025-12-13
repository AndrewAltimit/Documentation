"""
Diffusion Models

Implementation of diffusion models including score-based models, DDPM,
and advanced diffusion techniques.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreBasedDiffusion:
    """Score-based diffusion models with continuous time formulation"""

    def __init__(
        self, score_model: nn.Module, sigma_min: float = 0.01, sigma_max: float = 50.0
    ):
        self.score_model = score_model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Variance preserving (VP) SDE noise schedule"""
        # σ(t) = σ_min * (σ_max/σ_min)^t
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of p_t(x_t|x_0)"""
        std = self.noise_schedule(t)
        mean = x
        return mean, std

    def sde_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient f(x,t) for dx = f(x,t)dt + g(t)dw"""
        return torch.zeros_like(x)

    def sde_diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient g(t)"""
        sigma = self.noise_schedule(t)
        return torch.sqrt(
            2
            * torch.log(self.sigma_max / self.sigma_min)
            * sigma**2
            / (self.sigma_max / self.sigma_min) ** (2 * t)
        )

    def reverse_sde(
        self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Reverse-time SDE drift"""
        drift = self.sde_drift(x, t)
        diffusion = self.sde_diffusion(t)
        return drift - diffusion**2 * score

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Denoising score matching loss"""
        batch_size = x.shape[0]

        # Sample time uniformly
        t = torch.rand(batch_size, device=x.device)

        # Get marginal distribution parameters
        mean, std = self.marginal_prob(x, t)

        # Sample from marginal
        z = torch.randn_like(x)
        perturbed_x = mean + std[:, None, None, None] * z

        # Predict score
        score = self.score_model(perturbed_x, t)

        # Score matching loss
        loss = torch.mean(
            torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
        )
        return loss

    @torch.no_grad()
    def sample(
        self, shape: Tuple[int, ...], device: str = "cuda", num_steps: int = 1000
    ) -> torch.Tensor:
        """Sample using Predictor-Corrector sampler"""
        # Initialize from prior
        x = torch.randn(shape, device=device) * self.sigma_max

        # Time steps
        t = torch.linspace(1, 0, num_steps + 1, device=device)
        dt = -1 / num_steps

        for i in range(num_steps):
            # Current and next time
            t_curr = t[i] * torch.ones(shape[0], device=device)

            # Predictor step (Euler-Maruyama)
            score = self.score_model(x, t_curr)
            drift = self.reverse_sde(x, t_curr, score)
            diffusion = self.sde_diffusion(t_curr)

            x = x + drift * dt + diffusion * np.sqrt(-dt) * torch.randn_like(x)

            # Corrector step (Langevin dynamics)
            if i < num_steps - 1:
                score = self.score_model(x, t_curr)
                noise = torch.randn_like(x)
                grad_norm = torch.mean(torch.sum(score**2, dim=(1, 2, 3)))
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                step_size = 2 * (noise_norm / grad_norm) ** 2
                x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        return x


class DDPM:
    """Denoising Diffusion Probabilistic Models"""

    def __init__(
        self,
        model: nn.Module,
        T: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.model = model
        self.T = T

        # Linear variance schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process
        q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None, None
        ]

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for reverse process
        p(x_{t-1} | x_t)
        """
        # Predict noise
        epsilon_theta = self.model(x_t, t)

        # Compute mean
        beta_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None, None
        ]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]

        mean = sqrt_recip_alphas_t * (
            x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * epsilon_theta
        )

        # Compute variance
        posterior_variance_t = self.posterior_variance[t][:, None, None, None]

        return mean, posterior_variance_t

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t)
        """
        mean, variance = self.p_mean_variance(x_t, t)

        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float()[:, None, None, None]

        return mean + nonzero_mask * torch.sqrt(variance) * noise

    def loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Simplified loss: E_t,ε[||ε - ε_θ(x_t, t)||²]
        """
        batch_size = x_0.shape[0]

        # Sample timesteps
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], device: str = "cuda") -> torch.Tensor:
        """Generate samples using reverse diffusion"""
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device)
            x = self.p_sample(x, t_batch)

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, ...],
        device: str = "cuda",
        ddim_timesteps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Denoising Diffusion Implicit Models (DDIM) sampling"""
        # Select subset of timesteps
        c = self.T // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.T, c)))

        # Start from pure noise
        x = torch.randn(shape, device=device)

        for i in reversed(range(ddim_timesteps)):
            t = torch.full((shape[0],), ddim_timestep_seq[i], device=device)

            # Predict x_0
            epsilon_theta = self.model(x, t)

            alpha_t = self.alphas_cumprod[t][:, None, None, None]
            alpha_t_prev = (
                self.alphas_cumprod_prev[t][:, None, None, None]
                if i > 0
                else torch.ones_like(alpha_t)
            )

            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * epsilon_theta) / torch.sqrt(
                alpha_t
            )

            # Direction pointing to x_t
            sigma_t = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )

            # Compute x_{t-1}
            mean_pred = (
                torch.sqrt(alpha_t_prev) * x_0_pred
                + torch.sqrt(1 - alpha_t_prev - sigma_t**2) * epsilon_theta
            )

            noise = torch.randn_like(x) if i > 0 else 0
            x = mean_pred + sigma_t * noise

        return x


class LatentDiffusionModel(nn.Module):
    """Latent Diffusion Model with VAE encoder/decoder"""

    def __init__(
        self, vae: nn.Module, diffusion_model: nn.Module, scale_factor: float = 0.18215
    ):
        super().__init__()
        self.vae = vae
        self.diffusion_model = diffusion_model
        self.scale_factor = scale_factor

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space"""
        h = self.vae.encoder(x)
        moments = self.vae.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return self.scale_factor * posterior.sample()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        z = 1.0 / self.scale_factor * z
        h = self.vae.post_quant_conv(z)
        dec = self.vae.decoder(h)
        return dec

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through LDM"""
        # Encode to latent
        z = self.encode(x)

        # Add noise in latent space
        noise = torch.randn_like(z)
        noisy_z = self.diffusion_model.q_sample(z, t, noise)

        # Predict noise
        if conditioning is not None:
            predicted_noise = self.diffusion_model.model(noisy_z, t, **conditioning)
        else:
            predicted_noise = self.diffusion_model.model(noisy_z, t)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Sample from LDM"""
        # Sample in latent space
        latent_shape = (
            shape[0],
            4,
            shape[2] // 8,
            shape[3] // 8,
        )  # Assuming 8x downsampling

        if conditioning is not None:
            z_samples = self.diffusion_model.sample(latent_shape, device, conditioning)
        else:
            z_samples = self.diffusion_model.sample(latent_shape, device)

        # Decode to image space
        samples = self.decode(z_samples)
        return samples


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE"""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        else:
            return self.mean + self.std * torch.randn_like(self.mean)

    def kl(
        self, other: Optional["DiagonalGaussianDistribution"] = None
    ) -> torch.Tensor:
        if other is None:
            # KL(q||N(0,1))
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3]
            )
        else:
            # KL(q||p)
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )

    def nll(self, sample: torch.Tensor, dims: List[int] = [1, 2, 3]) -> torch.Tensor:
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )


class GuidedDiffusion:
    """Classifier and classifier-free guidance for diffusion models"""

    def __init__(self, diffusion_model: DDPM, classifier: Optional[nn.Module] = None):
        self.diffusion_model = diffusion_model
        self.classifier = classifier

    def classifier_guided_sample(
        self,
        shape: Tuple[int, ...],
        class_label: int,
        guidance_scale: float = 1.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Sample with classifier guidance"""
        if self.classifier is None:
            raise ValueError("Classifier required for classifier guidance")

        x = torch.randn(shape, device=device)

        for t in reversed(range(self.diffusion_model.T)):
            t_batch = torch.full((shape[0],), t, device=device)

            # Get model prediction
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                epsilon = self.diffusion_model.model(x, t_batch)

                # Get classifier gradient
                logits = self.classifier(x, t_batch)
                selected = logits[:, class_label]
                gradient = torch.autograd.grad(selected.sum(), x)[0]

            # Modify noise prediction with gradient
            sigma_t = self.diffusion_model.sqrt_one_minus_alphas_cumprod[t]
            epsilon = epsilon - guidance_scale * sigma_t * gradient

            # Sample with modified noise
            mean, variance = self.diffusion_model.p_mean_variance(x, t_batch)
            noise = torch.randn_like(x) if t > 0 else 0
            x = mean + torch.sqrt(variance) * noise

        return x

    def classifier_free_guidance(
        self,
        shape: Tuple[int, ...],
        conditioning: torch.Tensor,
        guidance_scale: float = 7.5,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Classifier-free guidance"""
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.diffusion_model.T)):
            t_batch = torch.full((shape[0],), t, device=device)

            # Conditional and unconditional predictions
            epsilon_cond = self.diffusion_model.model(x, t_batch, conditioning)
            epsilon_uncond = self.diffusion_model.model(x, t_batch, None)

            # Combine predictions
            epsilon = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)

            # Sample
            mean = self._get_mean_from_epsilon(x, t_batch, epsilon)
            variance = self.diffusion_model.posterior_variance[t]
            noise = torch.randn_like(x) if t > 0 else 0
            x = mean + torch.sqrt(variance) * noise

        return x

    def _get_mean_from_epsilon(
        self, x_t: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean from noise prediction"""
        beta_t = self.diffusion_model.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = (
            self.diffusion_model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        )
        sqrt_recip_alphas_t = self.diffusion_model.sqrt_recip_alphas[t][
            :, None, None, None
        ]

        return sqrt_recip_alphas_t * (
            x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * epsilon
        )
