import torch
from torch import nn
from models.StableDiffusionV1.VariationalAutoEncoder import VAE
from models.StableDiffusionV1.CLIPTransformer import CLIPTransformer
from models.StableDiffusionV1.UNET import UNET

class StableDiffusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        pass

