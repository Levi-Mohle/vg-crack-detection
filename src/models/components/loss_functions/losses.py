import numpy as np
import torch
import torch.nn as nn
# from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.functional import structural_similarity_index_measure as SSIM

class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x, device, reduction ='mean'):
        if reduction == 'none':
            loss = nn.functional.mse_loss(input=x_hat, target=x, reduction = reduction).mean(dim=(1,2,3))
        else:
            loss = nn.functional.mse_loss(input=x_hat, target=x, reduction = reduction)
        return loss

class SSIM_loss(nn.Module):
    def __init__(self):

        super().__init__()
        self.kernel_size = 5
        self.sigma = 1.5

    # def normalize(self, x):
    #     min = x.min(dim=0, keepdim=True)[0]
    #     max = x.max(dim=0, keepdim=True)[0]
    #     x_norm = (x - min) / (max - min + 1e-8)
    #     return x_norm
        
    # def forward(self, x, x_hat, device, reduction = 'elementwise_mean'): 
        
    #     ssim = SSIM(kernel_size= self.kernel_size,
    #                 sigma = self.sigma,
    #                 reduction = reduction,
    #                 ).to(device)
    #     return torch.abs(1 - ssim(self.normalize(x), self.normalize(x_hat)))
    def forward(self, x, x_hat, device, reduction = 'elementwise_mean'):
        return 1- SSIM(x, x_hat, reduction = reduction)

    
class NLL_Typicality_Loss(nn.Module):
    """Get the combined loss on NLL and typicality.
        
    """
    def __init__(
        self,
        k: int = 256,
        alpha: float = 1.0, 
    ) -> None:
        
        super(NLL_Typicality_Loss, self).__init__()
        self.k = k
        self.alpha = alpha

    def forward(self, x, z, sldj):
        # Regular Negative Log-likelihood calcuation
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        # Adding typicality term
        # mean_nll = nll.mean()
        bpd = nll* np.log2(np.exp(1)) / np.prod(z.shape[-3:])
        mean_bpd = bpd.mean()
        # gradient_norm = torch.flatten(z.grad[:,1, ...], start_dim=1).norm(dim=1, p=2).mean(dim=0)
        grad_input = torch.autograd.grad(outputs = nll, inputs=x, create_graph=True, retain_graph=True, only_inputs=True)
        gradient_norm = grad_input[0].norm(2)

        # Compose loss
        loss = mean_bpd + self.alpha * gradient_norm

        return loss
    
class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(
        self, 
        k=256
    ) -> None:
        
        super(RealNVPLoss, self).__init__()
        self.k = k

    def forward(self, x, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll