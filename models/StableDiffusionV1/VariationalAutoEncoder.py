import torch
from torch import nn
import yaml
from diffusers import AutoencoderKL

with open("Parameters.yaml", "r") as f:
    config = yaml.safe_load(f)["StableDiffusion"]["VAE"]
    encoder_config = yaml.safe_load(f)["StableDiffusion"]["VAE"]["encoder"]
    decoder_config = yaml.safe_load(f)["StableDiffusion"]["VAE"]["decoder"]

class ResNet(nn.Module):
    """
    This class is for transforming visual features by adding non-linearity
    so the model doesn't decay into a linear system.

    The input tensor is passed through a group normalisation to divide the
    amount of channels by the number of groups which normalises the channels
    per group and the spatial dimensions. The SiLU activation adds non-linearity.
    The 3x3 convolutions make connections between the spatial features in neighbouring
    groups. Since, the channels don't change in ResNet adding seperate input and output
    channels are unecessary but might provide some use in later SD projects
    The residual tensor is then passed through a convolution to match the channels
    of x and the output is then returned.

    Args:
        in_channels: The input channels into the ResNet block
        out_channels: The output channels into the ResNet block
        num_groups: How many groups the channels are divided into

    Returns:
        x: The tensor that contains all spatial information of neighbours

    Example Tensor:
        Input, x: (batch_size, in_channels, height, width)
        after groupnorm1, x:(batch_size, in_channels, height, width)
        after silu1, x:(batch_size, in_channels, height, width)
        after conv1, x:(batch_size, out_channels, height, width)
        after groupnorm2, x:(batch_size, out_channels, height, width)
        after silu2, x:(batch_size, out_channels, height, width)
        after conv12 x:(batch_size, out_channels, height, width)
        after residual, x::(batch_size, out_channels, height, width)
        Output, x:(batch_size, out_channels, height, width)
        The tensor doesnt't change shape if channels remain the same
        but the values inside are effected.
    """
    def __init__(self, in_channels:int, out_channels:int, num_groups:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groupnorm1 = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6
        )
        self.groupnorm2 = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_channels, eps=1e-6
        )
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(1,1), padding=0, stride=1
        )

    def forward(self, x:torch.Tensor):
        residual = x
        x = self.groupnorm1(x)
        x = self.silu1(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.silu2(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x = x + self.residual_conv(residual)
        x = x + residual
        return x


class SelfAttention(nn.Module):
    """
    Spatial self attention facilitates the information exchange between
    all tokens.

    The input tensor is passed through a group normalisation to normalise values
    in the feature map within groups of channels. The input tensor is
    then passed through a learnable query, key and value projection. The query
    and key tensors determine relationships between the spatial positions. Value
    contains all the information that is exchanged and how much is dependent on
    the attention scores. The spatial dimensions become height*width for comparisons
    between the spatial positions. 

    Args:
        num_groups: The number of groups that the channels are split into
        channels: The input channels
        x: The input tensor containing all spatial visual information

    Returns:
        x: Tensor containing all spatial information after and before self
        attention

    Example Tensor:
        Input, x:(batch_size, channels, height, width)
        after groupnorm, x:(batch_size, channels, height, width)
        after query_conv, query:(batch_size, channels, height, width)
        after key_conv: key:(batch_size, channels, height, width)
        after value_conv: value:(batch_size, channels, height, width)
        after reshape: query:(batch_size, height*width, channels)
        after reshape: key:(batch_size, channels, height*width)
        after reshape: value:(batch_size, channels, height*width)
        attn_scores:(batch_size, height*width, width*height)
        attn_output:(batch, channels, height*width)
        after reshape, attn_output:(batch, channels, height, width)
        Output, attn_scores:(batch, channels, height, width)
    """
    def __init__(self, num_groups:int, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.channels = channels
        self.query_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1,1), stride=1,
            padding=0
        )
        self.key_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1,1), stride=1,
            padding=0
        )
        self.value_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1,1), stride=1,
            padding=0
        )
        self.projection = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1,1), stride=1,
            padding=0
        )

    def forward(self, x:torch.Tensor):
        residual = x
        batch, _, height, width = x.shape
        x = self.groupnorm(x)
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        query = query.permute(0,2,3,1).flatten(start_dim=1,end_dim=2)
        key = key.flatten(start_dim=2,end_dim=3)
        value = value.flatten(start_dim=2,end_dim=3)

        attn_scores = query @ key.transpose(-1,-2)
        attn_scores = attn_scores / (self.channels ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_output = value @ attn_scores.transpose(-1,-2)

        attn_output = attn_output.reshape(batch, self.channels, height, width)
        return residual + self.projection(attn_output)


class Downsample(nn.Module):
    """
    Halves the amount of spatial features and increases the amount of channels
    so the features become more rich. This is used to compress the image and 
    eventually form the latent image used in the diffusion process

    The input tensor is passed through a convolution with a stride of two. The
    kernel then skips over half of the features to halve the amount of features
    returned. Output channels must be greater than input channels.

    Args:
        input_channels: The input tensor's channels
        output_channels: The desired output tensor's channels
        x: The input tensor

    Returns:
        x: The desired tensor with increased channels and decreased features

    Example Tensor:
        Input, x:(batch, input_channels, height, width)
        Output, x:(batch, output_channels, height/2, width/2)
    """
    def __init__(self, input_channels:int, output_channels:int):
        super().__init__()
        self.downsampling_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.downsampling_conv(x)
        return x

class Upsample(nn.Module):
    """
    Doubles the amount of spatial features and decreases the amount of channels.
    This is used to expand the image and form the final generated image.

    Using nearest-neighbour interpolation it increases the spatial resolution.
    New pixels are created by taking reference from the nearest original pixel so 
    thus increasing the amount of pixels per area increases the spatial resolution.
    The 3x3 convolution decreases the channels to mirror the downsamping path and 
    reduce computational load.

    Args:
        input_channels: The input tensor's channels
        output_channels: The desired output tensor's channels
        x: The input tensor
        scale_factor: The factor the the spatial dimension are scaled at

    Returns:
        x: The desired tensor with decreased channels and increased features

    Example Tensor:
        Input, x:(batch, input_channels, height, width)
        Output, x:(batch, output_channels, height*2, width*2)
    """
    def __init__(self, input_channels:int, output_channels:int, scale_factor:int=config["upsample_scale_factor"]):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.upsampling_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.upsample(x)
        x = self.upsampling_conv(x)
        return x

class Encoder(nn.Module):
    """
    Converts the image x into a latent representation of z to be used in the diffusion process

    The image is passed through several ResNet and Downsample blocks to transform the spatial features
    and add non-linearity to the weights and also to decrease the spatial dimensions in the feature
    map and increase the number of channels in the image. In the middle block, the self attention blocks
    determines the relationships and the information exchange between all features. The transformed
    image is passed then to a group normalisation to normal the feature map values in the groups per channel.
    The SiLU activation enchances the gradients flow and stability of the weights. The output is then projected
    into the number of latent channels and then returned.

    Args:
        input_channels: The encoder's input channels
        output_channels: The encoder's output channels
        rgb_channels: The number of channels in rgb
        num_groups: The number of groups that the channels are divided into

    Returns:
        An transformed image that has be downsized 
    """
    def __init__(self, input_channels:int=encoder_config["input_channels"],
                output_channels:int=encoder_config["output_channels"],
                rgb_channels:int=config["rgb_channels"],
                num_groups:int=config["num_groups"],
                num_downblocks:int=encoder_config["num_downblocks"],
                num_resnets:int=encoder_config["num_resnets"],
                num_downsamplers:int=encoder_config["num_downsamplers"]):
        super().__init__()
        self.num_downblocks = num_downblocks
        self.input_channels = input_channels
        self.input_channelsx2 = input_channels * 2
        self.input_channelsx4 = input_channels * 4
        self.input_conv = nn.Conv2d(
            in_channels=rgb_channels,
            out_channels=input_channels,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )
        self.down_block1 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels)
                for _ in range(num_resnets)
            ]
            +
            [
                Downsample(input_channels=self.input_channels, output_channels=self.input_channelsx2)
                for _ in range(num_downsamplers)
            ]
                
        )
        self.down_block2 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx2, out_channels=self.input_channelsx2)
                for _ in range(num_resnets)
            ]
            +
            [
                Downsample(input_channels=self.input_channelsx2, output_channels=self.input_channelsx4)
                for _ in range(num_downsamplers)
            ]
        )
        self.down_block3 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4)
                for _ in range(num_resnets)
            ]
            +
            [
                Downsample(input_channels=self.input_channelsx4, output_channels=self.input_channelsx4)
                for _ in range(num_downsamplers)
            ]
        )
        self.down_block4 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4)
                for _ in range(num_resnets)
            ]
        )
        self.middle_block = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4),
                SelfAttention(num_groups=num_groups, channels=self.input_channelsx4),
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4)
            ]
        )
        self.down_blocks = nn.ModuleList(
            [
                self.down_block1,
                self.down_block2,
                self.down_block3,
                self.down_block4
            ]
        )
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=self.input_channelsx4, eps=1e-6)
        self.silu = nn.SiLU()
        self.proj_conv = nn.Conv2d(
            in_channels=self.input_channelsx4, out_channels=output_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.input_conv(x)
        for i in range(self.num_downblocks):
            for block in self.down_blocks[i]:
                x = block(x)
        for block in self.middle_block:
            x = block(x)
        x = self.groupnorm(x)
        x = self.silu(x)
        x = self.proj_conv(x)
        return x


class Decoder(nn.Module):
    """
    Converts the latent representation z that has undergone denoising to the generated image x_hat 
    
    The latent representation passes through the middle block where the ResNet blocks transform
    the spatial features and add non-linearity similar to the encoder blocks. The self attention
    class is used to exchange positional information with all features. The input tensor then passes
    through several up blocks which contain the upsampling class. This increases the amount of 
    spatial features and decreases the number of channels. The transformed image is passed then to a
    group normalisation to normal the feature map values in the groups per channel. The SiLU activation
    enchances the gradients flow and stability of the weights. The output is then projected
    as rgb channels before being returned.

    Args:
        input_channels: The decoders's input channels
        output_channels: The decoders's output channels
        latent_channels: The amount of channels the latent representation has
        num_groups: The number of groups that the channels are divided into

    Returns:
        Generated image
    """
    def __init__(self, rgb_channels:int=config["rgb_channels"],
                latent_channels:int=config["latent_channels"],
                input_channels:int=decoder_config["input_channels"],
                num_groups:int=config["num_groups"],
                num_upblocks:int=decoder_config["num_upblocks"],
                num_resnets:int=decoder_config["num_resnets"],
                num_upsamplers:int=decoder_config["num_upsamplers"]):
        super().__init__()
        self.num_upblocks = num_upblocks
        self.input_channels = input_channels
        self.input_channels_2 = input_channels // 2
        self.input_channels_4 = input_channels // 4
        self.proj_conv = nn.Conv2d(
            in_channels=latent_channels, out_channels=input_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.middle_block = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels),
                SelfAttention(num_groups=num_groups, channels=input_channels),
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels)
            ]
        )
        self.up_block1 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels)
                for _ in range(num_resnets)
            ]
            +
            [
                Upsample(input_channels=input_channels, output_channels=self.input_channels_2)
                for _ in range(num_upsamplers)
            ]
        )
        self.up_block2 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channels_2, out_channels=self.input_channels_2)
                for _ in range(num_resnets)           
            ]
            +
            [
                Upsample(input_channels=self.input_channels_2, output_channels=self.input_channels_4)
                for _ in range(num_upsamplers)
            ]
        )
        self.up_block3 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channels_4, out_channels=self.input_channels_4)
                for _ in range(num_resnets)      
            ]
            +
            [
                Upsample(input_channels=self.input_channels_4, output_channels=self.input_channels_4)
                for _ in range(num_upsamplers)
            ]
        )
        self.up_block4 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channels_4, out_channels=self.input_channels_4)
                for _ in range(num_resnets)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                self.up_block1,
                self.up_block2,
                self.up_block3,
                self.up_block4
            ]
        )
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=self.input_channels_4, eps=1e-6)
        self.silu = nn.SiLU(inplace=True)
        self.output_conv = nn.Conv2d(
            in_channels=self.input_channels_4, out_channels=rgb_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.post_quant_conv(x)
        x = self.proj_conv(x)
        for block in self.middle_block:
            x = block(x)
        for i in range(self.num_upblocks):
            for block in self.up_blocks[i]:
                x = block(x)
        x = self.groupnorm(x)
        x = self.silu(x)
        x = self.output_conv(x)
        return x

class LatentDistribution(nn.Module):
    """
    This class creates the latent representation by applying noise to the mean
    and variance information of the input tensor

    The noise tensor is randomly generated from a tensor with similar shape to x.
    The code is made device-agnostic by applying the device from x to the noise.
    Mean and log variance are produced by chunking the input tensor. Standard deviation
    is applied and the latent representation is returned.

    Args:
        x: The input tensor from the encoder's output

    Returns:
        z: The latent representation

    Example Tensor:
        x: (batch_size, channels, height, width)
        z: (batch_size, channels, height, width)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        mean, log_variance = x.chunk(chunks=2, dim=1)
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        std = torch.exp(log_variance * 0.5)
        z = (noise * std) + mean
        return z

class VAE(nn.Module):
    """
    This class is the final Variational Autoencoder class that uses KL regularization.

    Based on the user's choice this can be used to encode or decode images. If encode
    is chosen then the image x is passed through the encoder, quantisation convolution
    to compress the 512 channels used in the encoder to 8 channels needed for latent
    distribution which produces the latent representation z. If decode is chosen then
    the image passes through a post quantisation convolution to for projection and then
    passes through the decoder to produce the final generate image.

    Args:
        encode: If true the VAE will encode the input
        decode: If true the VAE will decode the input

    Returns:
        Either a latent representation or generated image
    """
    def __init__(self, encode:bool, decode:bool):
        super().__init__()
        self.encode = encode
        self.decode = decode
        if self.encode:
            self.decode = False
        if self.decode:
            self.encode = False
        assert self.encode == self.decode, "Encode and Decode cannot be the value!"
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(
            in_channels=encoder_config["output_channels"], out_channels=2*config["embed_dim"],
            kernel_size=(1,1), stride=1, padding=0
        )
        self.post_quant_conv = nn.Conv2d(
            in_channels=config["embed_dim"], out_channels=config["latent_channels"],
            kernel_size=(1,1), stride=1, padding=0
        )
        self.latent_distribution = LatentDistribution()

    def forward(self, x:torch.Tensor):
        if self.encode: 
            x = self.encoder(x)
            x = self.quant_conv(x)
            x = self.latent_distribution(x)
        elif self.decode:
            x = self.post_quant_conv(x)
            x = self.decoder(x)
        return x

    def load_pretrained(self):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        with torch.no_grad():
            # ---------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------
            self.encoder.input_conv.weight.copy_(
                vae.encoder.conv_in.weight
            )
            self.encoder.input_conv.bias.copy_(
                vae.encoder.conv_in.bias
            )
            # ---------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------
            for down_block_idx in range(encoder_config["num_downblocks"]):
                for resnet_idx in range(encoder_config["num_resnets"]):
                    self.encoder.down_blocks[down_block_idx][resnet_idx].groupnorm1.weight.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].norm1.weight
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].groupnorm1.bias.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].norm1.bias
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].groupnorm2.weight.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].norm2.weight
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].groupnorm2.bias.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].norm2.bias
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].conv1.weight.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].conv1.weight
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].conv1.bias.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].conv1.bias
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].conv2.weight.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].conv2.weight
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].conv2.bias.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].conv2.bias
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].residual_conv.weight.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].conv_shortcut.weight
                    )
                    self.encoder.down_blocks[down_block_idx][resnet_idx].residual_conv.bias.copy_(
                        vae.encoder.down_blocks[down_block_idx].resnets[resnet_idx].conv_shortcut.bias
                    )
                for downsample_idx in range(encoder_config["num_downsamplers"]):
                    if down_block_idx != encoder_config["num_downblocks"]:
                        self.encoder.down_blocks[down_block_idx][downsample_idx].downsampling_conv.weight.copy_(
                            vae.encoder.down_blocks[down_block_idx].downsamplers[downsample_idx].conv.weight
                        )
                        self.encoder.down_blocks[down_block_idx][downsample_idx].downsampling_conv.bias.copy_(
                            vae.encoder.down_blocks[down_block_idx].downsamplers[downsample_idx].conv.bias
                        )
                    else: continue

            for resnet_idx in range(encoder_config["num_resnets"]):
                if resnet_idx == 1: idx = 1
                self.encoder.middle_block[resnet_idx+idx].groupnorm1.weight.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].norm1.weight
                )
                self.encoder.middle_block[resnet_idx+idx].groupnorm1.bias.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].norm1.bias
                )
                self.encoder.middle_block[resnet_idx+idx].groupnorm2.weight.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].norm2.weight
                )
                self.encoder.middle_block[resnet_idx+idx].groupnorm2.bias.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].norm2.bias
                )
                self.encoder.middle_block[resnet_idx+idx].conv1.weight.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].conv1.weight
                )
                self.encoder.middle_block[resnet_idx+idx].conv1.bias.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].conv1.bias
                )
                self.encoder.middle_block[resnet_idx+idx].conv2.weight.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].conv2.weight
                )
                self.encoder.middle_block[resnet_idx+idx].conv2.bias.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].conv2.bias
                )
                self.encoder.middle_block[resnet_idx+idx].residual_conv.weight.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].conv_shortcut.weight
                )
                self.encoder.middle_block[resnet_idx+idx].residual_conv.bias.copy_(
                    vae.encoder.mid_block.resnets[resnet_idx].conv_shortcut.bias
                )
            self.encoder.middle_block[1].groupnorm.weight.copy_(
                vae.encoder.mid_block.attentions[0].group_norm.weight
            )
            self.encoder.middle_block[1].groupnorm.bias.copy_(
                vae.encoder.mid_block.attentions[0].group_norm.bias
            )
            self.encoder.middle_block[1].query_conv.weight.copy_(
                vae.encoder.mid_block.attentions[0].to_q.weight
            )
            self.encoder.middle_block[1].query_conv.bias.copy_(
                vae.encoder.mid_block.attentions[0].to_q.bias
            )
            self.encoder.middle_block[1].key_conv.weight.copy_(
                vae.encoder.mid_block.attentions[0].to_k.weight
            )
            self.encoder.middle_block[1].key_conv.bias.copy_(
                vae.encoder.mid_block.attentions[0].to_k.bias
            )
            self.encoder.middle_block[1].value_conv.weight.copy_(
                vae.encoder.mid_block.attentions[0].to_v.weight
            )
            self.encoder.middle_block[1].value_conv.bias.copy_(
                vae.encoder.mid_block.attentions[0].to_v.bias
            )
            self.encoder.middle_block[1].projection.weight.copy_(
                vae.encoder.mid_block.attentions[0].to_out[0].weight
            )
            self.encoder.middle_block[1].projection.bias.copy_(
                vae.encoder.mid_block.attentions[0].to_out[0].bias
            )
            # ---------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------
            for up_block_idx in range(decoder_config["num_upblocks"]):
                for resnet_idx in range(decoder_config["num_resnets"]):
                    self.decoder.up_blocks[up_block_idx][resnet_idx].groupnorm1.weight.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].norm1.weight
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].groupnorm1.bias.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].norm1.bias
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].groupnorm2.weight.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].norm2.weight
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].groupnorm2.bias.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].norm2.bias
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].conv1.weight.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].conv1.weight
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].conv1.bias.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].conv1.bias
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].conv2.weight.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].conv2.weight
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].conv2.bias.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].conv2.bias
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].residual_conv.weight.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].conv_shortcut.weight
                    )
                    self.decoder.up_blocks[up_block_idx][resnet_idx].residual_conv.bias.copy_(
                        vae.decoder.up_blocks[up_block_idx].resnets[resnet_idx].conv_shortcut.bias
                    )
                for upsample_idx in range(decoder_config["num_upsamplers"]):
                    if up_block_idx != decoder_config["num_upblocks"]:
                        self.decoder.up_blocks[up_block_idx][upsample_idx].upsampling_conv.weight.copy_(
                            vae.decoder.up_blocks[up_block_idx].upsamplers[upsample_idx].conv.weight
                        )
                        self.decoder.up_blocks[up_block_idx][upsample_idx].upsampling_conv.bias.copy_(
                            vae.decoder.up_blocks[up_block_idx].upsamplers[upsample_idx].conv.bias
                        )
                    else: continue

            for resnet_idx in range(decoder_config["num_resnets"]):
                if resnet_idx == 1: idx = 1
                self.decoder.middle_block[resnet_idx+idx].groupnorm1.weight.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].norm1.weight
                )
                self.decoder.middle_block[resnet_idx+idx].groupnorm1.bias.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].norm1.bias
                )
                self.decoder.middle_block[resnet_idx+idx].groupnorm2.weight.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].norm2.weight
                )
                self.decoder.middle_block[resnet_idx+idx].groupnorm2.bias.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].norm2.bias
                )
                self.decoder.middle_block[resnet_idx+idx].conv1.weight.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].conv1.weight
                )
                self.decoder.middle_block[resnet_idx+idx].conv1.bias.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].conv1.bias
                )
                self.decoder.middle_block[resnet_idx+idx].conv2.weight.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].conv2.weight
                )
                self.decoder.middle_block[resnet_idx+idx].conv2.bias.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].conv2.bias
                )
                self.decoder.middle_block[resnet_idx+idx].residual_conv.weight.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].conv_shortcut.weight
                )
                self.decoder.middle_block[resnet_idx+idx].residual_conv.bias.copy_(
                    vae.decoder.mid_block.resnets[resnet_idx].conv_shortcut.bias
                )
            self.decoder.middle_block[1].groupnorm.weight.copy_(
                vae.decoder.mid_block.attentions[0].group_norm.weight
            )
            self.decoder.middle_block[1].groupnorm.bias.copy_(
                vae.decoder.mid_block.attentions[0].group_norm.bias
            )
            self.decoder.middle_block[1].query_conv.weight.copy_(
                vae.decoder.mid_block.attentions[0].to_q.weight
            )
            self.decoder.middle_block[1].query_conv.bias.copy_(
                vae.decoder.mid_block.attentions[0].to_q.bias
            )
            self.decoder.middle_block[1].key_conv.weight.copy_(
                vae.decoder.mid_block.attentions[0].to_k.weight
            )
            self.decoder.middle_block[1].key_conv.bias.copy_(
                vae.decoder.mid_block.attentions[0].to_k.bias
            )
            self.decoder.middle_block[1].value_conv.weight.copy_(
                vae.decoder.mid_block.attentions[0].to_v.weight
            )
            self.decoder.middle_block[1].value_conv.bias.copy_(
                vae.decoder.mid_block.attentions[0].to_v.bias
            )
            self.decoder.middle_block[1].projection.weight.copy_(
                vae.decoder.mid_block.attentions[0].to_out[0].weight
            )
            self.decoder.middle_block[1].projection.bias.copy_(
                vae.decoder.mid_block.attentions[0].to_out[0].bias
            )
            # ---------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------
            self.encoder.groupnorm.weight.copy_(
                vae.encoder.conv_norm_out.weight
            )
            self.encoder.groupnorm.bias.copy_(
                vae.encoder.conv_norm_out.bias
            )
            self.encoder.proj_conv.weight.copy_(
                vae.encoder.conv_out.weight
            )
            self.encoder.proj_conv.bias.copy_(
                vae.encoder.conv_out.bias
            )
            self.decoder.proj_conv.weight.copy_(
                vae.decoder.conv_in.weight
            )
            self.decoder.proj_conv.bias.copy_(
                vae.decoder.conv_in.bias
            )
            # ---------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------
            self.decoder.groupnorm.weight.copy_(
                vae.decoder.conv_norm_out.weight
            )
            self.decoder.groupnorm.bias.copy_(
                vae.decoder.conv_norm_out.bias
            )
            self.decoder.output_conv.weight.copy_(
                vae.decoder.conv_out.weight
            )
            self.decoder.output_conv.bias.copy_(
                vae.decoder.conv_out.bias
            )
            # ---------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------
            self.quant_conv.weight.copy_(
                vae.quant_conv.weight
            )
            self.quant_conv.bias.copy_(
                vae.quant_conv.bias
            )
            self.post_quant_conv.weight.copy_(
                vae.post_quant_conv.weight
            )
            self.post_quant_conv.bias.copy_(
                vae.post_quant_conv.bias
            )
        return self

