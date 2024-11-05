import numpy as np
import torch
import torch.nn as nn

class NLL_Typicality_loss(nn.Module):
    """Get the combined loss on NLL and typicality.
        
    """
    def __init__(self):
        super().__init__()

    def forward(self, ll, alpha, inputs):
        nll = -(ll)
        # mean_nll = nll.mean()
        bpd = nll* np.log2(np.exp(1)) / np.prod(inputs.shape[-3:])
        mean_bpd = bpd.mean()
        gradient_norm = torch.flatten(inputs.grad[:,1, ...], start_dim=1).norm(dim=1, p=2).mean(dim=0)
        loss = mean_bpd + alpha * gradient_norm

        return loss