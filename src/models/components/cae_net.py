import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 feature_maps: list,
                 latent_dim: int,
                 ):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_channels, feature_maps, latent_dim)
        self.decoder = Decoder(input_channels, feature_maps, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat

class Encoder(nn.Module):
    def __init__(self, input_channels, feature_maps, latent_dim):

        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(feature_maps[0], feature_maps[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(feature_maps[1], feature_maps[2], kernel_size=3, stride=2),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self,x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_channels, feature_maps, latent_dim):
        
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(feature_maps[2], feature_maps[2], kernel_size=3, stride=2),
            # nn.ReLU(),
            nn.ConvTranspose2d(feature_maps[1],feature_maps[1], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps[1], feature_maps[0], kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps[0], input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)
    
if __name__ == "__main__":
    _ = AutoEncoder()