import torch
from torch import nn
import yaml
from models.StableDiffusionV1.DDIMScheduler import DDIMSchedular
from models.StableDiffusionV1.CLIPTransformer import CLIPTransformer
from models.StableDiffusionV1.UNET import UNET
from models.StableDiffusionV1.VariationalAutoEncoder import VAE
from transformers import CLIPTokenizer
with open("Parameters.yaml","r") as f:
    sd_config = yaml.safe_load(f)["StableDiffusion"]

class StableDiffusion(nn.Module):
    def __init__(self,
                #  tokenizer,
                #  unet:torch.nn.Module,
                #  vae:torch.nn.Module,
                #  clip:torch.nn.Module,
                #  scheduler,
                device:torch.device,
                max_seq_len:int=sd_config["max_seq_len"],
                height:int=sd_config["height"],
                width:int=sd_config["width"],
                num_inference_steps:int=sd_config["num_inference_steps"],
                guidance_scale:float=sd_config["guidance_scale"],
                eta:float=sd_config["eta"],
                T:int=sd_config["T"],
                ):
        super().__init__()

        # self.tokenizer = tokenizer
        # self.unet = unet
        # self.vae = vae
        # self.clip = clip
        # self.scheduler = scheduler

        self.tokenizer = CLIPTokenizer()
        self.unet = UNET().load_pretrained()
        self.vae = VAE(decode=True).load_pretrained()
        self.clip = CLIPTransformer().load_pretrained()
        self.scheduler = DDIMSchedular()

        self.device = device
        self.max_seq_len = max_seq_len
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.T = T

    def forward(self,
                positive_prompt:str,
                negative_prompt:str="",
                ):
        
        if self.height or self.width or self.num_inference_steps or self.guidance_scale <= 0:
            raise ValueError("Image Dimension and Inference steps must be greater than zero")

        if not 0 <= self.eta <= 1:
            raise ValueError("Eta must be between 0 and 1")

        positive_tokens = self.tokenizer(
            text=positive_prompt,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        negative_tokens = self.tokenizer(
            text=negative_prompt,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        positive_context = self.clip(
            x=positive_tokens["input_ids"].to(self.device)
        )

        negative_context = self.clip(
            x=negative_tokens["input_ids"].to(self.device)
        )

        x_t = torch.randn(
            size=(1,4,64,64),
            device=self.device,
            dtype="float32"
        )

        delta_t = self.T / self.num_inference_steps
        timestep = torch.arange(self.T-delta_t,0,delta_t,device=self.device)
        prev_t = 1000

        for t in timestep:
            pred_noise = self.unet(
                x_t,positive_context,t
            )
            x_t = self.scheduler.step(
                prev_x=x_t,eta=self.eta,
                pred_noise=pred_noise,timestep=t,
                prev_timestep=prev_t
            )
            prev_t = t








        

