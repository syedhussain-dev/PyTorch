import torch
import math
from typing import Tuple
import yaml

with open("Parameters.yaml","r") as f:
    ddim_config = yaml.safe_load(f)["StableDiffusion"]["DDIMScheduler"]

class DDIMScheduler():
    """
    Implements the forward and reverse diffusion processes using a DDIM scheduler.

    The forward process progressively corrupts a clean latent with Gaussian noise.
    The noise variance at each timestep is controlled by beta_t, while alpha_t
    determines the signal scaling between consecutive timesteps. The cumulative
    product alpha_bar_t determines the relative contributions of the original
    clean latent and Gaussian noise.

    During reverse diffusion, the scheduler uses the UNet's predicted noise to
    estimate the original clean latent and construct a sample at a lower noise
    level. As denoising progresses, the clean-signal contribution increases
    while the noise contribution decreases.

    The reverse process supports two sampling modes:
        - Deterministic (eta = 0): No additional noise is introduced. Identical
            initial conditions and model predictions produce identical outputs.
        - Stochastic (eta > 0): Additional Gaussian noise is sampled during
            reverse diffusion, allowing different denoising trajectories.

    Scheduler tensors are transferred to the appropriate device and dtype
    when required. 
    """
    def __init__(self):
        self.T = ddim_config["T"]
        self.beta_start = ddim_config["beta_start"]
        self.beta_end = ddim_config["beta_end"]
        self.beta, self.alpha, self.alpha_bar = self.calcu_alpha()

    def step(self, prev_x:torch.Tensor, eta:float, pred_noise:torch.Tensor,timestep:int,prev_timestep:int) -> torch.Tensor:
        self.ensure_device_dtype(prev_x.device)
        if not 0 <= timestep < self.T:
            raise ValueError("Invalid timestep")

        if not 0 <= eta <= 1:
            raise ValueError("eta must be between 0 and 1")

        alpha_bar_t = self.alpha_bar[timestep]
        if timestep <= 0:
            alpha_bar_t_1 = torch.ones_like(alpha_bar_t)
        else:
            alpha_bar_t_1  = self.alpha_bar[prev_timestep]
        sigma_t = eta * torch.sqrt(((1-alpha_bar_t_1)/(1-alpha_bar_t))*(1-(alpha_bar_t/alpha_bar_t_1)))
        pred_x_0 = (prev_x - pred_noise*(torch.sqrt(1-alpha_bar_t))) / (torch.sqrt(alpha_bar_t))
        if eta > 0:
            z = torch.randn_like(prev_x)
            x = torch.sqrt(alpha_bar_t_1) * pred_x_0 + torch.sqrt(1-alpha_bar_t_1-sigma_t**2) * pred_noise + sigma_t*z
        else:
            x = torch.sqrt(alpha_bar_t_1) * pred_x_0 + torch.sqrt(1-alpha_bar_t_1) * pred_noise
        return x

    def add_noise(self, x_0:torch.Tensor, noise:torch.Tensor,timestep:int) -> torch.Tensor:
        self.ensure_device_dtype(x_0.device)
        alpha_bar_t = self.alpha_bar[timestep]
        z_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return z_t

    def calcu_alpha(self) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        timestep_array = torch.arange(0,self.T,1, dtype=torch.float32)
        beta = ((math.sqrt(self.beta_start) + ((timestep_array)/(self.T-1))*(math.sqrt(self.beta_end) - math.sqrt(self.beta_start))) ** 2)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        return (beta,alpha,alpha_bar)

    def ensure_device_dtype(self, device:torch.device, dtype:torch.dtype | None = None):
        if self.alpha_bar.device != device or (dtype is not None and self.alpha_bar.dtype != dtype):
            self.beta = self.beta.to(device=device,dtype=dtype)
            self.alpha = self.alpha.to(device=device,dtype=dtype)
            self.alpha_bar = self.alpha_bar.to(device=device,dtype=dtype)