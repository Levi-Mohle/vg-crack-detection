import torch.nn as nn

class DeepSVDD(nn.Module):
    def __init__(self,
                 in_channels: int,
                 rep_dim: int,
                 ):
        super(DeepSVDD, self).__init__()
        self.rep_dim = rep_dim
        self.encoder = Encoder(in_channels, rep_dim)

    def forward(self, x):
        z = self.encoder(x)

        return z

class Encoder(nn.Module):
    def __init__(self, in_channels, rep_dim):

        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=5),
            ConvBlock(32, 64, kernel_size=5),
            ConvBlock(64, 128, kernel_size=5),
            nn.Flatten(),
            nn.Linear(128*4*4, rep_dim, bias=False),
        )

    def forward(self,x):
        return self.encoder(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2),
            nn.BatchNorm2d(out_channels, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
        )

    def forward(self,x):
        return self.convblock(x)

    
if __name__ == "__main__":
    _ = DeepSVDD()