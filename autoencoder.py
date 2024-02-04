import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision


class AutoEncoder(nn.Module):
    def __init__(self, latent_size=512):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [16, 16, 16]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [32, 8, 8]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [64, 4, 4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [128, 2, 2]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, latent_size, 2),  # [latent_size, 1, 1]
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 128, 2),  # [128, 2, 2]
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # [64, 4, 4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # [32, 8, 8]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # [16, 16, 16]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                16, 3, 3, stride=2, padding=1, output_padding=1
            ),  # [3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
