import torch
import torch.nn.functional as F
from torch import nn
import os
from diffusers import UNet2DConditionModel
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir,"Parameters.yaml")

with open(yaml_path,"r") as f:
    unet_config = yaml.safe_load(f)["StableDiffusion"]["UNET"]
    vae_config = yaml.safe_load(f)["StableDiffusion"]["VAE"]

class TimestepEmbedding(nn.Module):
    """
    Converting the timestep to a 1D vector quantity for each image given in a batch

    A 1D vector of zeros is created with a batch number of rows and embedded dimension
    for the number of columns. The code is made device agnostic and converting to an 
    essential datatype. Then apply the frequency equation i: sin(t/10000^i/(d/2-s))
    and i+d/2: cos(t/10000^i/(d/2-s)). This tensor is then pass through a mlp projection
    or learnable projection to prepare it for the injection into the ResNet blocks

    Args:
        temb_channels: The embedded dimension
        timestep: The scalar we want to convert into a vector
        expanded_channels: The channels used in the mlp projection to add features and non-linearity

    Returns:
        a: The 1D timestep vector for each image

    Example:
        a: (batch_size, temb_channels)
        After for loop: (batch_size, temb_channels)
        After projection: (batch_size, 1280)
    """
    def __init__(self, temb_channels:int, out_channels:int):
        super().__init__()
        self.temb_channels = temb_channels
        assert self.temb_channels % 2 == 0, "Embedded Dimension should be even!"
        self.time_mlp_projection = nn.Sequential(
            nn.Linear(in_features=temb_channels, out_features=out_channels),
            nn.SiLU(),
            nn.Linear(in_features=out_channels,out_features=out_channels)
        )

    def vectorize(self, timesteps:torch.Tensor,
                  embedding_dim:int,
                  flip_sin_cos:bool=False,
                  max_period:int=10000,
                  downscale_freq_shift:float=1.0,
                  scale_factor:float=1.0):
        assert embedding_dim % 2 == 0, "Embedding dimension must be even"
        half_dim = embedding_dim // 2
        temb = timesteps / torch.pow(max_period,torch.arange(0,half_dim,1)/half_dim-downscale_freq_shift)
        temb = timesteps[:,None] * temb[None,:]
        temb *= scale_factor
        if flip_sin_cos:
            temb = torch.cat([torch.cos(temb),torch.sin(temb)],dim=-1)
        else:
            temb = torch.cat([torch.sin(temb),torch.cos(temb)],dim=-1)
        return temb

    def forward(self, timestep:torch.Tensor) -> torch.Tensor:
        temb = self.vectorize(timestep=timestep,embedding_dim=self.temb_channels)
        temb = self.time_mlp_projection(temb)
        return temb

class ResNet(nn.Module):
    """
    Adding the timestep vector to a latent representation of the image

    A projection and several convolutions, SiLU activations, and group 
    normalisations are initalised in the constructor. The latent image
    passes through the first group normalisation and activation, then
    pass through a convolution that refines features. The timestep 
    vector is projected from the expanded_channels in the mlp step to 
    the out_channels corresponding to the channels in the U-Net. The 
    vector is then reshaped to add two new dimensions and added 
    to the latent image. The latent images then pass through a second
    group normalisation and activation, then pass through a convolution 
    that does further refinement after timestep injection. The final 
    output of the ResNet block is also added to the initial latent
    image tensors for making the information learn from the original.

    Args:
        in_channels: The input channels of the ResNet block
        out_channels: The output channels of the ResNet block
        expanded_channels: The mlp channels
        num_groups: The amount of groups that pass through the group norm
        x: The original latent image tensor
        timestep_vector: The 2D timestep vector

    Returns:
        x: The latent image tensor which has timestep embeddings and original latent information

    Example:
        x: (batch_size, in_channels, height, width)
        after groupnorm1: (batch_size, in_channels, height, width)
        after silu1: (batch_size, in_channels, height, width)
        after conv1: (batch_size, out_channels, height, width)
        after timestep embedding: (batch_size, out_channels, height, width)
        after groupnorm2: (batch_size, out_channels, height, width)
        after silu2: (batch_size, out_channels, height, width)
        after conv2: (batch_size, out_channels, height, width)
        after residual addition: (batch_size, out_channels, height, width)
    """
    def __init__(self, in_channels:int, out_channels:int, expanded_channels:int, num_groups:int, resnet_eps:float):
        super().__init__()
        self.timestep_projection = nn.Linear(
            in_features=expanded_channels, out_features=out_channels
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(1,1), stride=1, padding=0
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.groupnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=resnet_eps)
        self.groupnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=resnet_eps)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x:torch.Tensor, temb:torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.groupnorm1(x)
        x = self.silu1(x)
        x = self.conv1(x)
        t = temb
        t = self.timestep_projection(t)
        t = t.reshape(t.shape[0],t.shape[1],1,1)
        x = x + t
        x = self.groupnorm2(x)
        x = self.silu2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x = x + self.residual_conv(residual)
        else:
            x = x + residual
        return x
        
class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.downsampling_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels,
            kernel_size=(3,3), stride=2, padding=1
        )

    def forward(self, x:torch.Tensor):
        x = self.downsampling_conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, scale_factor:int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.upsampling_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels,
            kernel_size=(3,3), stride=1, padding=0
        )

    def forward(self, x:torch.Tensor):
        x = self.upsample(x)
        x = self.upsampling_conv(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim:int, heads:int):
        super().__init__()
        assert embed_dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // heads
        self.q_projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.k_projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.v_projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.heads = heads

    def forward(self, x:torch.Tensor):
        batch, tokens, _ = x.shape
        query = self.q_projection(x)
        key = self.k_projection(x)
        value = self.v_projection(x)

        query = query.reshape(batch, tokens, self.heads, self.head_dim).permute(0,2,1,3)
        key = key.reshape(batch, tokens, self.heads, self.head_dim).permute(0,2,1,3)
        value = value.reshape(batch, tokens, self.heads, self.head_dim).permute(0,2,1,3)

        attn_output = F.scaled_dot_product_attention(query,key,value)
        return self.projection(attn_output)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, context_dim:int, heads:int, embed_dim:int):
        super().__init__()
        assert embed_dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // heads
        self.heads = heads
        self.q_projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.k_projection = nn.Linear(in_features=context_dim, out_features=embed_dim)
        self.v_projection = nn.Linear(in_features=context_dim, out_features=embed_dim)
        self.out_projection = nn.Conv2d(
            in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=(1,1), stride=1, padding=0
        )

    def forward(self, x:torch.Tensor, context:torch.Tensor):
        batch, channels, height, width = x.shape
        _, seq_len,_ = context.shape
        query = self.q_projection(x)
        query = x.flatten(start_dim=2, end_dim=3).reshape(batch, self.heads, self.head_dim, height*width).permute(0,1,3,2)
        key = self.k_projection(context)
        key = key.reshape(batch, seq_len, self.heads, self.head_dim).permute(0,2,1,3)
        value = self.v_projection(context)
        value = value.reshape(batch, seq_len, self.heads, self.head_dim).permute(0,2,1,3)

        attn_scores = query @ key.transpose(-1,-2)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1) @ value
        attn_output = attn_scores.transpose(2,3).reshape(batch, channels, height, width)
        return self.out_projection(attn_output)

class FeedForward(nn.Module):
    def __init__(self, channels:int, ff_expansion:int):
        super().__init__()
        self.linear_projection_1 = nn.Linear(in_features=channels, out_features=channels*2*ff_expansion)
        self.gelu = nn.GELU()
        self.linear_projection_2 = nn.Linear(in_features=channels*ff_expansion, out_features=channels)

    def forward(self, x:torch.Tensor):
        x = self.linear_projection_1(x)
        path1, path2 = x.chunk(2, dim=-1)
        path1 = self.gelu(path1)
        output = path1 * path2
        return self.linear_projection_2(output)

class BasicTransformerBlock(nn.Module):
    def __init__(self, normalisation_eps:float, context_dim:int, heads:int, ff_expansion:int, channels:int):
        super().__init__()
        self.norm1 = nn.LayerNorm(eps=normalisation_eps)
        self.attn1 = MultiHeadSelfAttention(embed_dim=channels, heads=heads)
        self.norm2 = nn.LayerNorm(eps=normalisation_eps)
        self.attn2 = MultiHeadCrossAttention(context_dim=context_dim, embed_dim=channels, heads=heads)
        self.norm3 = nn.LayerNorm(eps=normalisation_eps)
        self.ff = FeedForward(channels=channels, ff_expansion=ff_expansion)

    def forward(self, x:torch.Tensor):
        x = self.norm1(x)
        x = self.attn1(x)
        x = self.norm2(x)
        x = self.attn2(x)
        x = self.norm3(x)
        x = self.ff(x)
        return x

class Transformer2D(nn.Module):
    def __init__(self, normalisation_eps:float, channels:int, num_encoder_layers:int, num_groups:int, ff_expansion:int, heads:int, context_dim:int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=normalisation_eps)
        self.proj_in = nn.Conv2d(in_channels=channels, out_channels=channels,
                                 kernel_size=(3,3),stride=1, padding=1)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(normalisation_eps=normalisation_eps,
                                      context_dim=context_dim,
                                      channels=channels,
                                      ff_expansion=ff_expansion,
                                      heads=heads)
                for _ in range(num_encoder_layers)
            ]
        )
        self.proj_out = nn.Linear(in_features=channels, out_features=channels)

    def forward(self, x:torch.Tensor):
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.proj_out(x)
        return x

# U-NET BLOCKS ---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, input_channels:int,
                output_channels:int,
                resnet_input_channels:int,
                resnet_output_channels:int,
                num_groups:int=unet_config["num_groups"],
                eps:float=unet_config["normalisation_eps"],
                expanded_channels:int=unet_config["time_mlp_channels"],
                context_dim:int=unet_config["context_dim"],
                ff_expansion:int=unet_config["ff_expansion"],
                heads:int=unet_config["heads"],
                num_encoder_layers:int=unet_config["transformer2d_num_encoder_layers"],
                num_layers:int=unet_config["cross_attn_down_num_blocks"]):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResNet(in_channels=resnet_input_channels, out_channels=resnet_output_channels,
                expanded_channels=expanded_channels, num_groups=num_groups,
                resnet_eps=eps)
                for _ in range(num_layers)
            ]
        )
        self.transformers = nn.ModuleList(
            [Transformer2D(normalisation_eps=eps,context_dim=context_dim,channels=input_channels,
                num_encoder_layers=num_encoder_layers, ff_expansion=ff_expansion,
                heads=heads)
                for _ in range(num_layers)
            ]
        )
        self.downsample = Downsample(input_channels=input_channels,output_channels=output_channels)

    def forward(self, x:torch.Tensor, context:torch.Tensor, temb:torch.Tensor)-> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        for res_block, trans_block in zip(self.resnets,self.transformers):
            x = res_block(x,temb)
            x = trans_block(x,context)
        output_states = ()
        output_states += (x,)
        x = self.downsample(x)
        return x, output_states
        
class DownBlock2D(nn.Module):
    def __init__(self, input_channels:int,
                output_channels:int,
                resnet_input_channels:int,
                resnet_output_channels:int,
                expanded_channels:int=unet_config["time_mlp_channels"],
                eps:float=unet_config["normalisation_eps"],
                num_groups:int=unet_config["num_groups"],
                num_layers:int=unet_config["down_num_blocks"]):
        super().__init__()
        self.resnets = nn.ModuleList(
            [   
                ResNet(in_channels=resnet_input_channels, out_channels=resnet_output_channels,
                expanded_channels=expanded_channels, num_groups=num_groups,
                resnet_eps=eps)
                for _ in range(num_layers)
            ]
        )
        self.downsample = Downsample(input_channels=input_channels,output_channels=output_channels)

    def forward(self, x:torch.Tensor, temb:torch.Tensor)-> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        for res_block in self.resnets:
            x = res_block(x, temb)
        output_states = ()
        output_states += (x,)
        x = self.downsample(x)
        return x, output_states

class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self,
                resnet_input_channels:int,
                resnet_output_channels:int,
                num_groups:int=unet_config["num_groups"],
                eps:float=unet_config["normalisation_eps"],
                expanded_channels:int=unet_config["time_mlp_channels"],
                context_dim:int=unet_config["context_dim"],
                ff_expansion:int=unet_config["ff_expansion"],
                heads:int=unet_config["heads"],
                num_encoder_layers:int=unet_config["transformer2d_num_encoder_layers"]):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResNet(in_channels=resnet_input_channels, out_channels=resnet_output_channels,
                   expanded_channels=expanded_channels, num_groups=num_groups,
                   resnet_eps=eps),
            ResNet(in_channels=resnet_input_channels, out_channels=resnet_output_channels,
                    expanded_channels=expanded_channels, num_groups=num_groups,
                    resnet_eps=eps)
        ])
        self.transformer = Transformer2D(normalisation_eps=eps,context_dim=context_dim,channels=resnet_input_channels,
                                  num_encoder_layers=num_encoder_layers,num_groups=num_groups,
                                  ff_expansion=ff_expansion,heads=heads)

    def forward(self, x:torch.Tensor, context:torch.Tensor, temb:torch.Tensor)-> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.resnets[0](x,temb)
        x = self.transformer(x,context)
        x = self.resnets[1](x,temb)
        return x

class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, input_channels:int,
                output_channels:int,
                resnet_input_channels:int,
                resnet_output_channels:int,
                num_groups:int=unet_config["num_groups"],
                eps:float=unet_config["normalisation_eps"],
                expanded_channels:int=unet_config["time_mlp_channels"],
                context_dim:int=unet_config["context_dim"],
                upsample_scale_factor:int=unet_config["upsample_scale_factor"],
                ff_expansion:int=unet_config["ff_expansion"],
                heads:int=unet_config["heads"],
                num_encoder_layers:int=unet_config["transformer2d_num_encoder_layers"],
                num_layers:int=unet_config["cross_attn_up_num_blocks"]):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResNet(in_channels=resnet_input_channels, out_channels=resnet_output_channels,
                    expanded_channels=expanded_channels, num_groups=num_groups,
                    resnet_eps=eps)
                for _ in range(num_layers)
            ]
        )
        self.transformers = nn.ModuleList(
            [
                Transformer2D(normalisation_eps=eps,context_dim=context_dim,channels=input_channels,
                    num_encoder_layers=num_encoder_layers,num_groups=num_groups,
                    ff_expansion=ff_expansion,heads=heads)
                for _ in range(num_layers)
            ]
        )
        self.upsample = Upsample(input_channels=input_channels, output_channels=output_channels,
                                 scale_factor=upsample_scale_factor)

    def forward(self, x:torch.Tensor, output_states:torch.Tensor, context:torch.Tensor, temb:torch.Tensor):
        x += (output_states,)
        for res_block, trans_block in zip(self.resnets, self.transformers):
            x = res_block(x,temb)
            x = trans_block(x,context)
        x = self.upsample(x)
        return x

class UpBlock2D(nn.Module):
    def __init__(self, input_channels:int,
                output_channels:int,
                resnet_input_channels:int,
                resnet_output_channels:int,
                expanded_channels:int=unet_config["temb_channels"],
                eps:float=unet_config["normalisation_eps"],
                num_groups:int=unet_config["num_groups"],
                upsample_scale_factor:int=unet_config["upsample_scale_factor"],
                num_layers:int=unet_config["up_num_blocks"]):
        super().__init__()
        self.resnets = nn.ModuleList(
            [   
                ResNet(in_channels=resnet_input_channels, out_channels=resnet_output_channels,
                expanded_channels=expanded_channels, num_groups=num_groups,
                resnet_eps=eps)
                for _ in range(num_layers)
            ]
        )
        self.upsample = Upsample(input_channels=input_channels, output_channels=output_channels,
                                    scale_factor=upsample_scale_factor)
    def forward(self, x:torch.Tensor, output_states:torch.Tensor, temb:torch.Tensor):
        x += (output_states,)
        for res_block in self.resnets:
            x = res_block(x,temb)
        x = self.upsample(x)
        return x

class UNET(nn.Module):
    def __init__(self, latent_channels:int=vae_config["latent_channels"],
                 input_channels:int=unet_config["input_conv_channels"]):
        super().__init__()
        self.timestep_embedding = TimestepEmbedding()
        self.input_channels = input_channels # 320
        self.input_channelsx2 = input_channels * 2 # 640
        self.input_channelsx4 = input_channels * 4 # 1280
        self.input_conv = nn.Conv2d(
            in_channels=latent_channels, out_channels=input_channels,
            kernel_size=(1,1),stride=1,padding=0
        )
        self.down_blocks = nn.ModuleList([
            CrossAttnDownBlock2D(input_channels=self.input_channels,
                                output_channels=self.input_channels,
                                resnet_input_channels=self.input_channels,
                                resnet_output_channels=self.input_channels),

            CrossAttnDownBlock2D(input_channels=self.input_channels,
                                output_channels=self.input_channelsx2,
                                resnet_input_channels=self.input_channels,
                                resnet_output_channels=self.input_channels),

            CrossAttnDownBlock2D(input_channels=self.input_channelsx2,
                                output_channels=self.input_channelsx4,
                                resnet_input_channels=self.input_channelsx2,
                                resnet_output_channels=self.input_channelsx2),

            DownBlock2D(input_channels=self.input_channelsx4,
                        output_channels=self.input_channelsx4,
                        resnet_input_channels=self.input_channelsx4,
                        resnet_output_channels=self.input_channelsx4)
        ])
        self.mid_block = UNetMidBlock2DCrossAttn(
            resnet_input_channels=self.input_channelsx4,
            resnet_output_channels=self.input_channelsx4
        )
        self.up_blocks = nn.ModuleList([
            UpBlock2D(input_channels=self.input_channelsx4,
                    output_channels=self.input_channelsx4,
                    resnet_input_channels=self.input_channelsx4,
                    resnet_output_channels=self.input_channelsx4),

            CrossAttnUpBlock2D(input_channels=self.input_channelsx4,
                                output_channels=self.input_channelsx4,
                                resnet_input_channels=self.input_channelsx4,
                                resnet_output_channels=self.input_channelsx4),

            CrossAttnUpBlock2D(input_channels=self.input_channelsx4,
                                output_channels=self.input_channelsx2,
                                resnet_input_channels=self.input_channelsx4,
                                resnet_output_channels=self.input_channelsx4),

            CrossAttnUpBlock2D(input_channels=self.input_channelsx2,
                                output_channels=self.input_channels,
                                resnet_input_channels=self.input_channelsx2,
                                resnet_output_channels=self.input_channelsx2)
        ])
        self.conv_norm_out = nn.GroupNorm(num_groups=unet_config["num_groups"],
                                          num_channels=self.input_channels)
        self.conv_act = nn.SiLU()
        self.output_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=latent_channels,
            kernel_size=(1,1), stride=1, padding=0
        )

    @torch.inference_mode()
    def forward(self, x:torch.Tensor, context:torch.Tensor, timestep:torch.Tensor):
        assert len(timestep.shape) == 1
        temb = self.timestep_embedding(timestep)
        x = self.input_conv(x)
        x, output0 = self.down_blocks[0](x, context, temb)
        x, output1 = self.down_blocks[1](x, context, temb)
        x, output2 = self.down_blocks[2](x, context, temb)
        x, output3 = self.down_blocks[3](x, temb)
        x = self.mid_block(x, context, temb)
        x = self.up_blocks[0](x, output3, temb)
        x = self.up_blocks[1](x, output2, context, temb)
        x = self.up_blocks[2](x, output1, context, temb)
        x = self.up_blocks[3](x, output0, context, temb)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.output_conv(x)
        return x

    def load_pretrained(self):
        unet = UNet2DConditionModel().from_pretrained("runwayml/stable-diffusion-v1-5")
        with torch.no_grad():
            self.input_conv.weight.copy_(
                unet.conv_in.weight
            )
            self.input_conv.bias.copy_(
                unet.conv_in.bias
            )
            self.output_conv.weight.copy_(
                unet.conv_out.weight
            )
            self.output_conv.bias.copy_(
                unet.conv_out.bias
            )
            self.conv_norm_out.weight.copy_(
                unet.conv_norm_out.weight
            )
            self.conv_norm_out.bias.copy_(
                unet.conv_norm_out.bias
            )
            self.timestep_embedding.time_mlp_projection[0].weight.copy_(
                unet.time_embedding.linear_1.weight
            )
            self.timestep_embedding.time_mlp_projection[0].bias.copy_(
                unet.time_embedding.linear_1.bias
            )
            self.timestep_embedding.time_mlp_projection[2].weight.copy_(
                unet.time_embedding.linear_2.weight
            )
            self.timestep_embedding.time_mlp_projection[2].bias.copy_(
                unet.time_embedding.linear_2.bias
            )
            # Down Blocks ---------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------
            for downblock_idx in range(len(self.down_blocks)):
                for resnet_idx in range(unet_config["cross_attn_down_num_blocks"]):
                    self.down_blocks[downblock_idx].resnets[resnet_idx].groupnorm1.weight.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].norm1.weight
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].groupnorm1.bias.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].norm1.bias
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].groupnorm2.weight.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].norm2.weight
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].groupnorm2.bias.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].norm2.bias
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].conv1.weight.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].conv1.weight
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].conv1.bias.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].conv1.bias
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].conv2.weight.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].conv2.weight
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].conv2.bias.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].conv2.bias
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].timestep_projection.weight.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].time_emb_proj.weight
                    )
                    self.down_blocks[downblock_idx].resnets[resnet_idx].timestep_projection.bias.copy_(
                        unet.down_blocks[downblock_idx].resnets[resnet_idx].time_emb_proj.bias
                    )
                if downblock_idx is not 4:
                    for trans_idx in range(unet_config["cross_attn_down_num_blocks"]):
                        self.down_blocks[downblock_idx].transformers[trans_idx].norm.weight.copy_(
                            unet.down_blocks[downblock_idx].attentions[trans_idx].norm.weight
                        )
                        self.down_blocks[downblock_idx].transformers[trans_idx].norm.bias.copy_(
                            unet.down_blocks[downblock_idx].attentions[trans_idx].norm.bias
                        )
                        self.down_blocks[downblock_idx].transformers[trans_idx].proj_in.weight.copy_(
                            unet.down_blocks[downblock_idx].attentions[trans_idx].proj_in.weight
                        )
                        self.down_blocks[downblock_idx].transformers[trans_idx].proj_in.bias.copy_(
                            unet.down_blocks[downblock_idx].attentions[trans_idx].proj_in.bias
                        )
                        self.down_blocks[downblock_idx].transformers[trans_idx].proj_out.weight.copy_(
                            unet.down_blocks[downblock_idx].attentions[trans_idx].proj_out.weight
                        )
                        self.down_blocks[downblock_idx].transformers[trans_idx].proj_out.bias.copy_(
                            unet.down_blocks[downblock_idx].attentions[trans_idx].proj_out.bias
                        )
                        for trans_block_idx in range(unet_config["transformer2d_num_encoder_layers"]):
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm1.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm1.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm1.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm1.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm2.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm2.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm2.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm2.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm3.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm3.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm3.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm3.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.q_projection.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_q.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.q_projection.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_q.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.k_projection.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_k.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.k_projection.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_k.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.v_projection.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_v.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.v_projection.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_v.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.out_projection.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_out[0].weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.out_projection.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_out[0].bias
                            )

                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.q_projection.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_q.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.q_projection.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_q.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.k_projection.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_k.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.k_projection.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_k.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.v_projection.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_v.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.v_projection.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_v.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.out_projection.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_out[0].weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.out_projection.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_out[0].bias
                            )

                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].ff.linear_projection_1.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].ff.net[0].proj.weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].ff.linear_projection_1.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].ff.net[0].proj.bias
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].ff.linear_projection_2.weight.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].ff.net[2].weight
                            )
                            self.down_blocks[downblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].ff.linear_projection_2.bias.copy_(
                                unet.down_blocks[downblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].ff.net[2].bias
                            )
                self.down_blocks[downblock_idx].downsample.downsampling_conv.weight.copy_(
                    unet.down_blocks[downblock_idx].downsamplers[0].conv.weight
                )
                self.down_blocks[downblock_idx].downsample.downsampling_conv.bias.copy_(
                    unet.down_blocks[downblock_idx].downsamplers[0].conv.bias
                )
            # Up Blocks ---------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------
            for upblock_idx in range(len(self.up_blocks)):
                for resnet_idx in range(unet_config["cross_attn_up_num_blocks"]):
                    self.up_blocks[upblock_idx].resnets[resnet_idx].groupnorm1.weight.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].norm1.weight
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].groupnorm1.bias.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].norm1.bias
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].groupnorm2.weight.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].norm2.weight
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].groupnorm2.bias.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].norm2.bias
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].conv1.weight.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].conv1.weight
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].conv1.bias.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].conv1.bias
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].conv2.weight.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].conv2.weight
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].conv2.bias.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].conv2.bias
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].timestep_projection.weight.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].time_emb_proj.weight
                    )
                    self.up_blocks[upblock_idx].resnets[resnet_idx].timestep_projection.bias.copy_(
                        unet.up_blocks[upblock_idx].resnets[resnet_idx].time_emb_proj.bias
                    )
                if upblock_idx is not 1:
                    for trans_idx in range(unet_config["cross_attn_up_num_blocks"]):
                        self.up_blocks[upblock_idx].transformers[trans_idx].norm.weight.copy_(
                            unet.up_blocks[upblock_idx].attentions[trans_idx].norm.weight
                        )
                        self.up_blocks[upblock_idx].transformers[trans_idx].norm.bias.copy_(
                            unet.up_blocks[upblock_idx].attentions[trans_idx].norm.bias
                        )
                        self.up_blocks[upblock_idx].transformers[trans_idx].proj_in.weight.copy_(
                            unet.up_blocks[upblock_idx].attentions[trans_idx].proj_in.weight
                        )
                        self.up_blocks[upblock_idx].transformers[trans_idx].proj_in.bias.copy_(
                            unet.up_blocks[upblock_idx].attentions[trans_idx].proj_in.bias
                        )
                        self.up_blocks[upblock_idx].transformers[trans_idx].proj_out.weight.copy_(
                            unet.up_blocks[upblock_idx].attentions[trans_idx].proj_out.weight
                        )
                        self.up_blocks[upblock_idx].transformers[trans_idx].proj_out.bias.copy_(
                            unet.up_blocks[upblock_idx].attentions[trans_idx].proj_out.bias
                        )
                        for trans_block_idx in range(unet_config["transformer2d_num_encoder_layers"]):
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm1.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm1.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm1.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm1.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm2.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm2.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm2.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm2.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm3.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm3.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].norm3.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].norm3.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.q_projection.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_q.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.q_projection.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_q.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.k_projection.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_k.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.k_projection.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_k.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.v_projection.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_v.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.v_projection.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_v.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.out_projection.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_out[0].weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn1.out_projection.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn1.to_out[0].bias
                            )

                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.q_projection.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_q.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.q_projection.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_q.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.k_projection.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_k.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.k_projection.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_k.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.v_projection.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_v.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.v_projection.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_v.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.out_projection.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_out[0].weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].attn2.out_projection.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].attn2.to_out[0].bias
                            )

                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].ff.linear_projection_1.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].ff.net[0].proj.weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].ff.linear_projection_1.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].ff.net[0].proj.bias
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].ff.linear_projection_2.weight.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].ff.net[2].weight
                            )
                            self.up_blocks[upblock_idx].transformers[trans_idx].transformer_blocks[trans_block_idx].ff.linear_projection_2.bias.copy_(
                                unet.up_blocks[upblock_idx].attentions[trans_idx].transformer_blocks[trans_block_idx].ff.net[2].bias
                            )
                self.up_blocks[upblock_idx].upsample.upsampling_conv.weight.copy_(
                    unet.up_blocks[upblock_idx].upsamplers[0].conv.weight
                )
                self.up_blocks[upblock_idx].upsample.upsampling_conv.bias.copy_(
                        unet.up_blocks[upblock_idx].upsamplers[0].conv.bias
                    )
        return self