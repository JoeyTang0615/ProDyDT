import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional

class Tools:
    def gather(self, consts: torch.Tensor, t: torch.Tensor):
        # Gathers constants according to timesteps 't'
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1)  # Adjust dimensions for broadcasting

class DiffusionModel():
    def __init__(self, eps_model, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma = self.beta
        self.tools = Tools()
        self.device = device

    # Forward diffusion: q(xt | x0)
    def q_xt_x0(self, x0_coords: torch.Tensor, t: torch.Tensor):
        alpha_bar_t = self.tools.gather(self.alpha_bar, t).reshape(-1, 1, 1)
        mean_coords = (alpha_bar_t ** 0.5) * x0_coords
        var_coords = 1 - alpha_bar_t
        return mean_coords, var_coords

    # Forward diffusion: sample xt ~ q(xt | x0)
    def q_sample(self, x0_coords: torch.Tensor, t: torch.Tensor, eps_coords: Optional[torch.Tensor] = None):
        if eps_coords is None:
            eps_coords = torch.randn_like(x0_coords)
        mean_coords, var_coords = self.q_xt_x0(x0_coords, t)
        xt_coords = mean_coords + (var_coords ** 0.5) * eps_coords
        return xt_coords

    # Reverse diffusion: sample xt-1 ~ p(xt-1 | xt)
    def p_sample(self, xt_coords: torch.Tensor, esm_features: torch.Tensor, t: torch.Tensor,guide_info1 = None, guide_info2 = None):
        # Predict noise (eps_hat) for coordinates using the DiT model
        eps_hat_coords = self.eps_model(xt_coords, esm_features, t, guide_info1, guide_info2)
        alpha_bar_t = self.tools.gather(self.alpha_bar, t).reshape(-1, 1, 1)
        alpha_t = self.tools.gather(self.alpha, t).reshape(-1, 1, 1)

        eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** 0.5

        mean_coords = 1 / (alpha_t ** 0.5) * (xt_coords - eps_coef * eps_hat_coords)

        var = self.tools.gather(self.sigma, t).reshape(-1, 1, 1)

        eps_coords = torch.randn_like(xt_coords)

        xtm1_coords = mean_coords + (var ** 0.5) * eps_coords

        return xtm1_coords

    # Loss function: computes the MSE loss between the predicted noise and actual noise
    def loss(self, x0_coords: torch.Tensor, esm_features: torch.Tensor, guide_info1 = None, guide_info2 = None,name = None,noise_coords: Optional[torch.Tensor] = None):
        batch_size = x0_coords.shape[0]

        # Sample random timesteps for each protein conformation
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0_coords.device, dtype=torch.long)

        if noise_coords is None:
            noise_coords = torch.randn_like(x0_coords)

        # Sample noisy versions of x0_coords
        xt_coords = self.q_sample(x0_coords, t, eps_coords=noise_coords)

        # Predict noise (eps_hat) using the model
        eps_hat_coords = self.eps_model(xt_coords, esm_features, t, guide_info1, guide_info2, name)

        # Compute MSE loss between actual noise and predicted noise
        loss_coords = F.mse_loss(noise_coords, eps_hat_coords)

        
        return loss_coords
    
    
