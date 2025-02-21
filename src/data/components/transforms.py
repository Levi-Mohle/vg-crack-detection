import torch
import numpy as np
import cv2
from typing import Any, Sequence
import random
import torchvision.transforms.functional as TF
import torchvision as TV
from scipy.ndimage import gaussian_filter
import torchvision.transforms.v2 as transforms
# from .loaders import ImageData
import skimage

#################################################################################
# Support functions
#################################################################################
# Convert images from 0-1 to 0-255 (integers)
def discretize_255(sample):
    return (sample * 255).to(torch.int32)

def ToTensor(x):
    return torch.tensor(x, dtype=torch.float32)

def normalize_rgb(x):
    """
    Function to transform height data from 8bit [0, 255] to 32float [-1, 1]
    """
    x = x / 255
    return x * 2 - 1

def normalize_height(x):
    """
    Function to transform height data from 16bit [0, 65535] to 32float [0, 1]
    """
    return x / 2**16

def normalize_idv(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
def normalize_height_idv(x):
    # to float32 is necessary due to subtracting not possible for uint16
    x = x.to(torch.float)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def rescale_diffuser_height(x):
    """
    Function to transform height data from 16bit [0, 65535] to 32float [-1, 1]
    """
    x = x / 2**16
    return x * 2 - 1

def rescale_diffuser(x):
    # to float32 is necessary due to subtracting not possible for uint16
    return x * 2 - 1

def rescale_diffuser_idv(x):
    # to float32 is necessary due to subtracting not possible for uint16
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x * 2 - 1

def rescale_diffuser_height_idv(x):
    x = x / 2**16
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x * 2 - 1

def inverse_norm(x):
     x = (x+1)/2
     return x 

#################################################################################
# Transforms
#################################################################################

# General normalization (between [0,1])
def normalize_0_1():
    transform = transforms.Compose([transforms.ToTensor(),
                                        ])
    return transform

def normalize_0_1_grayscale_idv():
    transform = transforms.Compose([ToTensor,
                                    transforms.Grayscale(),
                                    normalize_idv])
    return transform

def normalize_height_0_1():
    transform = transforms.Compose([ToTensor,
                                    normalize_height,])
    return transform

def normalize_height_0_1_idv():
    transform = transforms.Compose([ToTensor,
                                    normalize_height_idv,])
    return transform

# Normalization for Diffusers (between [-1,1]

def diffuser_to_grayscale():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Grayscale(),
                                    rescale_diffuser,
                                        ])
    return transform

def diffuser_to_grayscale_idv():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Grayscale(),
                                    rescale_diffuser_idv,
                                        ])
    return transform



def diffuser_normalize():
    transform = transforms.Compose([transforms.ToTensor(),
                                    rescale_diffuser,
                                        ])
    return transform

def diffuser_normalize_height():
    transform = transforms.Compose([transforms.ToTensor(),
                                    rescale_diffuser_height,
                                        ])
    return transform

def diffuser_normalize_height_idv():
    transform = transforms.Compose([transforms.ToTensor(),
                                    rescale_diffuser_height_idv,
                                        ])
    return transform

# Revert normalization

def revert_normalize_rgb():
    transform = transforms.Compose([transforms.ToTensor(),
                                    inverse_norm,
                                    transforms.Grayscale()
                                        ])
    return transform

def revert_normalize_height():
    transform = transforms.Compose([transforms.ToTensor(),
                                    inverse_norm,
                                        ])
    return transform

class Augmentation(transforms.Transform):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, rgb, height):
        rgb     = torch.tensor(rgb)
        height  = torch.tensor(height)

        if random.random() < self.p:
            rgb     = TF.hflip(rgb)
            height  = TF.hflip(height)

        if random.random() < self.p:
            rgb     = TF.vflip(rgb)
            height  = TF.vflip(height)

        return rgb, height

class CNNTransform(transforms.Transform):
    def __init__(self) -> None:
        super().__init__()
        self.rgb_transform    = normalize_0_1_grayscale_idv()
        self.height_transform = normalize_height_0_1_idv()

    def forward(self, rgb, height):
        rgb     = self.rgb_transform(rgb)
        height  = self.height_transform(height)

        return rgb, height
