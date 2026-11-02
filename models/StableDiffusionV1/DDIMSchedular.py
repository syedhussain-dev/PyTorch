import torch
from torch import nn, sqrt

class DDIMSchedular():
    def __init__(self):
        pass

    def step(self, x_t:torch.Tensor):
        pass

    def add_noise(self, x_0:torch.Tensor, epsilon:torch.Tensor):
        timestep_array = torch.arange(0,1000,1, dtype=torch.float32)
        alpha_t = torch.ones(size=timestep_array.shape)
        alpha_t[:] = 1 - ((sqrt(0.00085) + (timestep_array[:]/999)*(sqrt(0.012) - sqrt(0.00085))) ** 2)
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        z_t = sqrt(alpha_bar_t[:]) * x_0 + sqrt(1 - alpha_bar_t[:]) * epsilon
        return z_t