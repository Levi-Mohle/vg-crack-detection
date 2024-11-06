import torch
import torch.nn as nn
import numpy as np
import sys
from models.components.blocks.covflow_blocks import CouplingLayer, GatedConvNet, Dequantization, InvConv2d, AffineCouplingSdl
from models.components.utils.covflow_utils import create_checkerboard_mask

sys.path.append(r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\src")

num_coupling_layers = 2

x = torch.randn((32,3,28,28))
ldj = torch.zeros(x.shape[0])

batch_size, img_channels, img_height, img_width = x.shape

flow_layers = []
flow_layers += [Dequantization(quants=np.power(2, 16))]
for j in range(num_coupling_layers):
    flow_layers += [InvConv2d(in_channel=img_channels)]
    for i in range(2):
        flow_layers += [CouplingLayer( \
                        network=GatedConvNet(c_in=2*img_channels, c_out=2*img_channels, c_hidden=32, num_layers=4), 
                        mask = create_checkerboard_mask(h=img_height, w=img_width, invert=(i%2==1)), 
                        c_in=img_channels, 
                        is_conditioned=True)
                        ]  

flow_layers += [AffineCouplingSdl()]

flows = nn.ModuleList(flow_layers)

def encode(flows, imgs):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, torch.zeros(imgs.shape[0])
        for flow in flows:
    
            if type(flow).__name__ == 'CouplingLayer':
                if flow.is_conditioned == True:
                    z, ldj = flow(z, ldj, orig_img = imgs[:, :, ...], reverse=False)
                else:
                    z, ldj = flow(z, ldj, reverse=False)
            elif type(flow).__name__ == 'AffineCouplingSdl':
                z, ldj = flow(z, ldj, imgs = imgs[:, :, ...], reverse=False)
            else:
                z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

z, ldj = encode(flows, x)

print("done!")