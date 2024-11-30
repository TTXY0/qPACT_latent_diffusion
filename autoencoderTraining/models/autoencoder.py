import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import numpy as np
# import matplotlib.pyplot as plt
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from packaging import version # ONLY CHANGE
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.save_hyperparameters(ddconfig)
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        
        # self.pre_layer = torch.nn.Conv2d(1, 3, kernel_size=4, stride=4, padding=0, bias=False)
        # self.upsample = torch.nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4, stride=4, padding=0, bias=False)
        #Consider overlap in convolution
        # Two 2x downsampling layers
        self.pre_layer_1 = torch.nn.Conv2d(1, 3, kernel_size=2, stride=2, padding=0, bias=False)
        self.pre_layer_2 = torch.nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0, bias=False)

        #  upsampling layers
        self.upsample_1 = torch.nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0, bias=False)
        self.upsample_2 = torch.nn.ConvTranspose2d(3, 1, kernel_size=2, stride=2, padding=0, bias=False)

        
        with torch.no_grad(): 
            # self.pre_layer.weight.data.fill_(1/16)
            # self.upsample.weight.data.fill_(1/3)
            
            # for 1.3.3.3
            # self.pre_layer_1.weight.data.fill_(1/4)
            # self.pre_layer_2.weight.data.fill_(1/12)
            
            # self.upsample_1.weight.data.fill_(1/4)
            # self.upsample_2.weight.data.fill_(1/3)
            
            # for 1.32.32.3
            self.pre_layer_1.weight.data.fill_(1/4)
            self.pre_layer_2.weight.data.fill_(1/128) # 32 by 2 by 2
            
            self.upsample_1.weight.data.fill_(1/3) 
            self.upsample_2.weight.data.fill_(1/32)

        
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def preprocess(self, x):
        #print(f"Range of x before pre_layer_1: min = {x.min().item()}, max = {x.max().item()}")
        
        x = self.pre_layer_1(x)
        
        #print(f"Range of x after pre_layer_1: min = {x.min().item()}, max = {x.max().item()}")
        x = self.pre_layer_2(x)
        
        #print(f"Range of x after pre_layer_2: min = {x.min().item()}, max = {x.max().item()}")

        return x

    def post_process(self, x):
        #print(f"Range of x before upsample_1: min = {x.min().item()}, max = {x.max().item()}")
        
        x = self.upsample_1(x)
        
        #print(f"Range of x after upsample_1: min = {x.min().item()}, max = {x.max().item()}")

        x = self.upsample_2(x)
        #print(f"Range of x after upsample_2: min = {x.min().item()}, max = {x.max().item()}")

        return x

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        #print(f"Range of x after self.encoder: min = {h.min().item()}, max = {h.max().item()}")
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        #print(posterior.sample().shape)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.post_process(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def print_tensor_statistics(self, tensor, name="Tensor"):
        print(f"Statistics for {name}:")
        print(f"Mean: {tensor.mean().item()}")
        print(f"Standard Deviation: {tensor.std().item()}")
        print(f"Minimum Value: {tensor.min().item()}")
        print(f"Maximum Value: {tensor.max().item()}")
        print(f"Median: {tensor.median().item()}")
        print(f"Number of NaN Values: {torch.isnan(tensor).sum().item()}")
        print(f"Number of Inf Values: {torch.isinf(tensor).sum().item()}")
        print("")

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)  # forward
        
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        if reconstructions.shape[1] == 1:
            reconstructions = reconstructions.repeat(1, 3, 1, 1)
        loss, log_dict_ae = self.loss(
            inputs, 
            reconstructions, 
            posterior, 
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val"
            )
        #self.log("train/rec_loss", loss[3], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/MS_SSIM_loss", loss[1], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/pixel_loss", loss[2], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss[0]

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        
        if inputs.shape[1] == 1:  # Grayscale
            inputs = inputs.repeat(1, 3, 1, 1)
        if reconstructions.shape[1] == 1:  # Grayscale
            reconstructions = reconstructions.repeat(1, 3, 1, 1)

        loss, log_dict_ae = self.loss(
            inputs, 
            reconstructions, 
            posterior, 
            self.global_step, 
            last_layer=self.get_last_layer(), 
            split="val"
            )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        return self.log_dict



    def configure_optimizers(self):
        lr = self.learning_rate

        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                list(self.decoder.parameters()) +
                                list(self.quant_conv.parameters()) +
                                list(self.post_quant_conv.parameters()) +
                                list(self.pre_layer_1.parameters()) +         
                                list(self.pre_layer_2.parameters()) +    
                                list(self.upsample_1.parameters()) +    
                                list(self.upsample_2.parameters()),    
                                # list(self.conv_rgb_gray.parameters()),        
                                lr=lr, betas=(0.5, 0.9))
        # self.log("learning_rate", lr, prog_bar=True)
        return opt_ae

    def get_last_layer(self):
        return self.upsample_2.weight
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


# class IdentityFirstStage(torch.nn.Module):
#     def __init__(self, *args, vq_interface=False, **kwargs):
#         self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
#         super().__init__()

#     def encode(self, x, *args, **kwargs):
#         return x

#     def decode(self, x, *args, **kwargs):
#         return x

#     def quantize(self, x, *args, **kwargs):
#         if self.vq_interface:
#             return x, None, [None, None, None]
#         return x

#     def forward(self, x, *args, **kwargs):
#         return x
