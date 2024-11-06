import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from ssim import MS_SSIM

class MSSSIM_Loss(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=0, disc_weight=0,
                 perceptual_weight=0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.logvar = nn.Parameter(torch.ones(size=()) * -5.0)
        self.msssim_loss_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss ##### 
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        
    def get_partials(self, tensor):
        # partial derivatives
        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1] 
        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]

        # Pad to maintain original size
        dx = F.pad(dx, (0, 1, 0, 0)) 
        dy = F.pad(dy, (0, 0, 0, 1)) 

        return dx, dy

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            #nll_loss = nll_loss / 10000000
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        msssim_loss_weight = 1e6
        pixel_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        msssim_loss = 1 - self.msssim_loss_fn(inputs.contiguous(), reconstructions.contiguous())
        input_partial1, input_partial2 = self.get_partials(inputs)
        recon_partial1, recon_partial2 = self.get_partials(reconstructions)
        gradient_loss =  (torch.abs(input_partial1 - recon_partial1) + torch.abs(input_partial2 - recon_partial2))/.1 # 
        rec_loss = pixel_loss + msssim_loss * msssim_loss_weight
        
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        
        # Just for printing ... not used 
        gradient_loss = torch.sum(gradient_loss) / gradient_loss.shape[0]
        pixel_loss = torch.sum(pixel_loss) / pixel_loss.shape[0]
        
        loss = weighted_nll_loss + self.kl_weight * kl_loss
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                # "{}/d_weight".format(split): d_weight.detach(),
                # "{}/disc_factor".format(split): torch.tensor(disc_factor),
                # "{}/g_loss".format(split): g_loss.detach().mean(),
                }
        return [loss, msssim_loss.item() * msssim_loss_weight, pixel_loss], log
