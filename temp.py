import torch
from diffusers import (
    AutoencoderKL,
)
from diffusers.models import AutoencoderKL
from torch import nn


vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2', subfolder='vae')


class DepthVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2', subfolder='vae')
        # self.vae.decoder.conv_out = nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        for para in self.vae.parameters():
            para.requires_grad = False

        for param in self.vae.decoder.conv_out.parameters():
            param.requires_grad = True

        rgb_latent_scale_factor = 0.18215
        depth_latent_scale_factor = 0.18215

        self.rgb_latent_scale_factor = rgb_latent_scale_factor
        self.depth_latent_scale_factor = depth_latent_scale_factor

    def __replace_conv_out(self):
        self.vae.decoder.conv_out = nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # init weights
        nn.init.xavier_uniform_(self.vae.decoder.conv_out.weight)
        nn.init.zeros_(self.vae.decoder.conv_out.bias)

    def forward(self, x, mode='E'):
        if mode == 'E':
            latent = self.encode_rgb(x)

            return latent
        elif mode == 'ED':
            latent = self.encode_rgb(x)

            return self.decode_rgb(latent)
        elif mode == 'D':
            return self.decode_rgb(x)
        else:
            raise ValueError("Invalid mode value, should be 'E', 'ED', or 'D'.")

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            rgb_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        rgb_latent = rgb_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(rgb_latent)
        rgb = self.vae.decoder(z)
        return rgb

