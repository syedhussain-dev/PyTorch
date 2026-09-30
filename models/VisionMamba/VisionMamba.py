import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import yaml
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir,"Parameters.yaml")

with open(yaml_path,"r") as f:
	config = yaml.safe_load(f)["VisionMamba"]

class PatchEmbedding(nn.Module):
  """
  Splits an image into non-overlapping patches and projects each patch into
  an embedding vector using a Conv2D layer.

  The resulting patch embeddings are flattened and transposed into the
  format expected by a Vision Transformer.

  Args:
      patch_size: Height and width of each square image patch.
      in_channels: Number of input image channels (e.g. 3 for RGB).
      embed_dim: Dimension of the embedding vector produced for each patch.

  Returns:
      Tensor of shape (batch_size, num_patches, embed_dim).

  Example:
      Input:             (1, 3, 224, 224)
      After Conv2D:      (1, 768, 14, 14)
      After flatten():   (1, 768, 196)
      After transpose(): (1, 196, 768)
  """
  def __init__(self, patch_size, in_channels, embed_dim):
      super().__init__()
      self.create_patches = nn.Conv2d(
          in_channels=in_channels,
          out_channels=embed_dim,
          kernel_size=(patch_size, patch_size),
          stride=patch_size
      )
  def forward(self, x):
      x = self.create_patches(x)
      x = x.flatten(start_dim=2)
      x = x.transpose(1, 2)
      return x

class Discretization(nn.Module):
  """
  Converts a continuous system to a discrete system to process tokens

  dh/dt = Ah + Bx is the continous system which evolves with time but
  we use the discrete system h_t = A_bar * h_t-1 - B_bar * x_t to pass
  each token through it so h evolves with tokens. 
  """
  def __init__(self,ssm_dim:int, expand_dim:int):
    super().__init__()
    self.A = nn.Parameter(torch.rand((expand_dim,ssm_dim)))
    self.ssm_dim = ssm_dim

  def forward(self, delta:torch.Tensor, B:torch.Tensor):
    batch, _, expand_dim = delta.shape
    A_bar = torch.zeros(size=[batch,1,expand_dim,self.ssm_dim],device=delta.device)
    B_bar = torch.zeros(size=[batch,1,expand_dim,self.ssm_dim],device=delta.device)
    delta_A = delta.unsqueeze(dim=-1) * self.A.unsqueeze(dim=0).unsqueeze(dim=0)
    A_bar = torch.exp(delta_A)
    B_bar = ((A_bar - 1) * delta.unsqueeze(dim=-1) * B.unsqueeze(dim=2)) / delta_A
    return A_bar, B_bar

class SelectiveSSM(nn.Module):
  def __init__(self, embed_dim:int, ssm_dim:int, expand_dim:int, backward_pass:bool):
    super().__init__()
    self.B_projection = nn.Linear(expand_dim,ssm_dim)
    self.C_projection = nn.Linear(expand_dim,ssm_dim)
    self.delta_projection = nn.Linear(expand_dim,expand_dim)
    self.ssm_dim = ssm_dim
    self.embed_dim = embed_dim
    self.expand_dim = expand_dim
    self.discretization = Discretization(ssm_dim=ssm_dim,expand_dim=expand_dim)
    self.backward_pass = backward_pass

  def forward(self,x:torch.Tensor):
    batch,num_tokens,_ = x.shape
    outputs = []
    B = self.B_projection(x)
    C = self.C_projection(x)
    delta = F.softplus(self.delta_projection(x))
    residual_h = torch.zeros(size=(batch,self.expand_dim,self.ssm_dim),device=x.device)
    if self.backward_pass:
      B, C, delta = torch.flip(B,dims=[1]), torch.flip(C,dims=[1]), torch.flip(delta,dims=[1])
    for i in range(num_tokens):
      A_bar,B_bar = self.discretization(delta[:,i:i+1,:],B[:,i:i+1,:])
      h_t = (A_bar.squeeze(dim=1) * residual_h) + (B_bar.squeeze(dim=1) * x[:,i:i+1,:].squeeze(dim=1).unsqueeze(dim=-1))
      y_t = torch.sum(C[:,i:i+1,:] * h_t,dim=-1)
      outputs.append(y_t)
      residual_h = h_t
    return torch.stack(outputs,dim=1)

class VisionMambaEncoder(nn.Module):
  def __init__(self, embed_dim, ssm_dim, expand_dim):
    super().__init__()
    self.norm = nn.LayerNorm(embed_dim)
    self.X_projection = nn.Linear(in_features=embed_dim,out_features=expand_dim)
    self.Z_projection = nn.Linear(in_features=embed_dim,out_features=expand_dim)
    self.Y_projection = nn.Linear(in_features=expand_dim, out_features=embed_dim)

    self.conv_layer_1 = nn.Conv1d(in_channels=expand_dim, out_channels=expand_dim,kernel_size=3,padding=1,stride=1)
    self.conv_layer_2 = nn.Conv1d(in_channels=expand_dim, out_channels=expand_dim,kernel_size=3,padding=1,stride=1)

    self.forward_ssm = SelectiveSSM(
      embed_dim=embed_dim,
      ssm_dim=ssm_dim,
      expand_dim=expand_dim,
      backward_pass=False
    )

    self.backward_ssm = SelectiveSSM(
      embed_dim=expand_dim,
      ssm_dim=ssm_dim,
      expand_dim=expand_dim,
      backward_pass=True
    )

    self.embed_dim = embed_dim
    self.expand_dim = expand_dim
    self.ssm_dim = ssm_dim

  def forward(self, y):
    residual = y
    y = self.norm(y)
    x = self.X_projection(y)
    z = self.Z_projection(y)

    x_forward = self.conv_layer_1(x.transpose(1,2))
    x_forward = x_forward.transpose(1,2)

    x_backward = self.conv_layer_2(x.transpose(1,2))
    x_backward = x_backward.transpose(1,2)

    y_backward = self.backward_ssm(x_backward)
    y_forward = self.forward_ssm(x_forward)    
    
    y = y_forward * F.silu(z) + y_backward * F.silu(z)
    y = self.Y_projection(y)
    y = y + residual
    return y

class VisionMamba(nn.Module):
  def __init__(self, image_size:int=config["image_size"],
              patch_size:int=config["patch_size"],
              in_channels:int=config["in_channels"],
              embed_dim:int=config["embed_dim"],
              num_classes:int=config["num_classes"],
              num_encoder_layers:int=config["num_encoder_layers"],
              expand_dim:int=config["expand_dim"],
              ssm_dim:int=config["ssm_dim"],
              gradient_checkpointing:bool=config["gradient_checkpointing"]):
    super().__init__()

    self.gradient_checkpointing = gradient_checkpointing
    assert image_size % patch_size == 0
    self.num_patch = (image_size ** 2) // (patch_size ** 2)

    self.patch_embedding = PatchEmbedding(
      patch_size=patch_size,
      in_channels=in_channels,
      embed_dim=embed_dim
    )

    self.cls_token = nn.Parameter(
      torch.randn(1,1,embed_dim)
    )

    self.position_embedding = nn.Parameter(
      torch.randn(1, self.num_patch+1, embed_dim)
    )

    self.encoder_blocks = nn.ModuleList(
      [
        VisionMambaEncoder(
          embed_dim=embed_dim,
          ssm_dim=ssm_dim,
          expand_dim=expand_dim
        )
        for _ in range(num_encoder_layers)
      ]
    )

    self.mlp_head = nn.Linear(
      in_features=embed_dim,
      out_features=num_classes
    )

  def forward(self, x):
    x = self.patch_embedding(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + self.position_embedding
    for block in self.encoder_blocks:
      if self.gradient_checkpointing:
        x = checkpoint(block, x, use_reentrant=False)
      else:
        x = block(x)
    x = x.mean(dim=1)
    x = self.mlp_head(x)
    return x