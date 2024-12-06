import torch
import numpy as np
import cv2
from typing import Any, Sequence
import random
import torchvision.transforms.functional as TF
import torchvision as TV
from scipy.ndimage import gaussian_filter
from torchvision.transforms import transforms
# from .loaders import ImageData
import skimage

#################################################################################
# Support functions
#################################################################################
# Convert images from 0-1 to 0-255 (integers)
def discretize_255(sample):
    return (sample * 255).to(torch.int32)

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

def rescale_diffuser_height(x):
    """
    Function to transform height data from 16bit [0, 65535] to 32float [-1, 1]
    """
    x = x / 2**16
    return x * 2 - 1

def rescale_diffuser(sample):
    # to float32 is necessary due to subtracting not possible for uint16
    return sample * 2 - 1

#################################################################################
# Transforms
#################################################################################

def normalize_0_1():
    transform = transforms.Compose([transforms.ToTensor(),
                                        ])
    return transform

def diffuser_to_greyscale():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Grayscale(),
                                    rescale_diffuser,
                                        ])
    return transform

def normalize_height_0_1():
    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize_height,])
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

