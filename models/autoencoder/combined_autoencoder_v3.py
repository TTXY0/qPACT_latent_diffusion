# This loss function has only the KL divergence and MSE of the latent space

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import numpy as np
from packaging import version # ONLY CHANGE
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from torch import nn
from losses.ssim import MS_SSIM

class image_autoencoder(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 latent_ae_config,
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
        
        # Two 2x downsampling layers
        self.pre_layer_1 = nn.Conv2d(1, 3, kernel_size=2, stride=2, padding=0, bias=False)
        self.pre_layer_2 = nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0, bias=False)

        #  upsampling layers
        self.upsample_1 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0, bias=False)
        self.upsample_2 = nn.ConvTranspose2d(3, 1, kernel_size=2, stride=2, padding=0, bias=False)

        
        with torch.no_grad(): 
            self.pre_layer_1.weight.data.fill_(1/4)
            self.pre_layer_2.weight.data.fill_(1/128) # 32 by 2 by 2
            
            self.upsample_1.weight.data.fill_(1/3) 
            self.upsample_2.weight.data.fill_(1/32)

        
        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
        self.latent_ae = latent_autoencoder(**latent_ae_config) 

        # Freeze parameters in original autoencoder
        for param in self.parameters():
            param.requires_grad = False
        for param in self.latent_ae.parameters():
            param.requires_grad = True  # only train latent_autoencoder
            
        self.msssim_loss_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3)
    
    def preprocess(self, x):
        x = self.pre_layer_1(x)
        x = self.pre_layer_2(x)
        return x

    def post_process(self, x):      
        x = self.upsample_1(x)
        x = self.upsample_2(x)
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
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z[0])
        dec = self.decoder(z)
        dec = self.post_process(dec)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        # pass through inner autoencoder
        z_recon = self.latent_ae(z)
        
            
        dec = self.decode(z_recon)
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
        
        with torch.no_grad():
            posterior = self.encode(inputs)
            z_original = posterior.sample()  # Original latent vector
        
        # pass through latent autoencoder 
        z_recon, latent_posterior = self.latent_ae(z_original) 
    
        latent_mse = F.mse_loss(z_original, z_recon)
        
        # KL divergence
        kl_loss = latent_posterior.kl().mean()
        

        total_loss = latent_mse + kl_loss
        
        # Logging
        self.log("train/latent_mse", latent_mse, prog_bar=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        
        with torch.no_grad():
            posterior = self.encode(inputs)
            z_original = posterior.sample()
            
            z_recon, latent_posterior = self.latent_ae(z_original)
        
        latent_mse = F.mse_loss(z_original, z_recon)
        kl_loss = latent_posterior.kl().mean()
        total_val_loss = latent_mse + kl_loss
        
        self.log("val/latent_mse", latent_mse, prog_bar=True)
        self.log("val/kl_loss", kl_loss, prog_bar=True)
        self.log("val/total_loss", total_val_loss, prog_bar=True)
        
        return {"val_loss": total_val_loss}



    def configure_optimizers(self):
        lr = self.learning_rate

        # opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
        #                         list(self.decoder.parameters()) +
        #                         list(self.quant_conv.parameters()) +
        #                         list(self.post_quant_conv.parameters()) +
        #                         list(self.pre_layer_1.parameters()) +         
        #                         list(self.pre_layer_2.parameters()) +    
        #                         list(self.upsample_1.parameters()) +    
        #                         list(self.upsample_2.parameters()),     
        #                         lr=lr, betas=(0.5, 0.9))
        
        # def configure_optimizers(self):
        return torch.optim.Adam(self.latent_ae.parameters(), lr=lr)
        #return opt_ae

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

class latent_autoencoder(nn.Module):
    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, z):
        posterior = self.encode(z)
        z_recon = self.decode(posterior.sample())
        return z_recon, posterior