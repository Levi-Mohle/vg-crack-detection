import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

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

class LatentCenterVector(nn.Module):
    def __init__(self, rep_dim):

        super(LatentCenterVector, self).__init__()
        self.c = torch.nn.Parameter(torch.randn(rep_dim))

        
class DeeperSVDD(nn.Module):
    def __init__(self,
                 in_channels : int,
                 pretrained : bool
                 ):
        super(DeeperSVDD, self).__init__()

        # Determine to use pretrained network
        if pretrained:
            self.resnet = resnet34(weights=ResNet34_Weights)
        else:
            self.resnet = resnet34()

        original_conv1 =self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size= original_conv1.kernel_size,
            stride= original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        if pretrained:
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3, :, :] = original_conv1.weight
                self.resnet.conv1.weight[:, 3:, :, :] = original_conv1.weight.mean(dim=1, keepdim=True) 

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.flatten = nn.Flatten()

    def forward(self, x):
        features = self.resnet(x)

        flattened_features = self.flatten(features)

        return flattened_features
        
    
if __name__ == "__main__":
    _ = DeepSVDD()