"""
The archiecture for beta-VAE 
@alsolra 
"""

import torch 
from torch import nn 


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim)

    # def buildEncoder(self, latent_dim):
    #     encoder = nn.Sequential(

    #         nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=2, padding=1),
    #         nn.ELU(),

    #         nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
    #         nn.ELU(),

    #         nn.ConstantPad3d((0, 1, 0, 0), 0),
    #         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
    #         nn.ELU(),

    #         nn.ConstantPad3d((0, 0, 0, 1), 0),
    #         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
    #         nn.ELU(),

    #         nn.ConstantPad3d((0, 1, 0, 0), 0),
    #         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
    #         nn.ELU(),

    #         nn.ConstantPad3d((0, 0, 0, 1), 0),
    #         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
    #         nn.ELU(),

    #         nn.Flatten(start_dim=1, end_dim=-1),

    #         nn.Linear(2560, 256),
    #         nn.ELU(),

    #         nn.Linear(256, latent_dim * 2),
    #     )
    #     return encoder


    # def buildDecoder(self, latent_dim):
    #     decoder = nn.Sequential(

    #         nn.Linear(latent_dim, 256),
    #         nn.ELU(),

    #         nn.Linear(256, 256 * 5 * 2),
    #         nn.ELU(),

    #         nn.Unflatten(dim=1, unflattened_size=(256, 2, 5)),

    #         nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.ConstantPad2d((0, 0, 0, -1), 0),
    #         nn.ELU(),

    #         nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.ConstantPad2d((0, -1, 0, 0), 0),
    #         nn.ELU(),

    #         nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.ConstantPad2d((0, 0, 0, -1), 0),
    #         nn.ELU(),

    #         nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.ConstantPad2d((0, -1, 0, 0), 0),
    #         nn.ELU(),

    #         nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.ConstantPad2d((0, 0, 0, 0), 0),
    #         nn.ELU(),

    #         nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1),

    #     )
    #     return decoder
    
    def buildEncoder(self, latent_dim):
        encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 2x2
            nn.ELU(),
            nn.Flatten(start_dim=1, end_dim=-1),  # 256*2*2 = 1024
            nn.Linear(256*8*8, 256), # 1024 16384, PDEBench for low resolution: 1024, 256
            nn.ELU(),
            nn.Linear(256, latent_dim * 2),
        )
        return encoder

    def buildDecoder(self, latent_dim):
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256*8*8), # 1024 16384, PDEBench for low resolution: 1024, 256
            nn.ELU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 8, 8)),  # 256x2x2 or 8x8
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128
        )
        return decoder
    
    # this is for the cylinder data
    # def buildEncoder(self, latent_dim):
    #     return nn.Sequential(
    #         # ↓ same six conv‐downsamples ↓
    #         nn.Conv2d(3,  8, 3, stride=2, padding=1), # →161×65
    #         nn.ELU(),
    #         nn.Conv2d(8, 16, 3, stride=2, padding=1), # → 81×33
    #         nn.ELU(),
    #         nn.Conv2d(16,32, 3, stride=2, padding=1), # → 41×17
    #         nn.ELU(),
    #         nn.Conv2d(32,64, 3, stride=2, padding=1), # → 21× 9
    #         nn.ELU(),
    #         nn.Conv2d(64,128,3, stride=2, padding=1), # → 11× 5
    #         nn.ELU(),
    #         nn.Conv2d(128,256,3,stride=2, padding=1), # →  6× 3
    #         nn.ELU(),
    #         nn.Flatten(1),                             # 256*6*3 = 4608
    #         nn.Linear(4608, 256),
    #         nn.ELU(),
    #         nn.Linear(256, latent_dim * 2),
    #     )
    
    # def buildDecoder(self, latent_dim):
    #     return nn.Sequential(
    #         nn.Linear(latent_dim, 256),
    #         nn.ELU(),
    #         nn.Linear(256, 4608),              # 256*6*3
    #         nn.ELU(),
    #         nn.Unflatten(1, (256, 6, 3)),      # recover the 256×6×3 feature map

    #         # now six ConvTranspose2d to exactly reverse the encoder
    #         nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=0),  # 6→11
    #         nn.ELU(),
    #         nn.ConvTranspose2d(128, 64,  3, stride=2, padding=1, output_padding=0),  # 11→21
    #         nn.ELU(),
    #         nn.ConvTranspose2d(64,  32,  3, stride=2, padding=1, output_padding=0),  # 21→41
    #         nn.ELU(),
    #         nn.ConvTranspose2d(32,  16,  3, stride=2, padding=1, output_padding=0),  # 41→81
    #         nn.ELU(),
    #         nn.ConvTranspose2d(16,   8,  3, stride=2, padding=1, output_padding=0),  # 81→161
    #         nn.ELU(),
    #         nn.ConvTranspose2d(8,    3,  3, stride=2, padding=1, output_padding=0),  # 161→321
    #     )

    def sample(self, mean, logvariance):
        """
        Implementing reparameterlisation trick 
        """

        std = torch.exp(0.5 * logvariance)
        epsilon = torch.rand_like(std)

        return mean + epsilon*std

    def forward(self, data):
        mean_logvariance = self.encoder(data)

        mean, logvariance = torch.chunk(mean_logvariance, 2, dim=1)
        
        z = self.sample(mean, logvariance)

        reconstruction = self.decoder(z)

        return reconstruction, mean, logvariance
