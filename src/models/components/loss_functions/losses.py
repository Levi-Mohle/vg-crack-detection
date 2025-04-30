import numpy as np
import torch
import torch.nn as nn
# from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class CustomCosineSimilarity(nn.Module):
    def __init__(self,
                 DLlambda: float):
        super().__init__()
        self.DLlambda = DLlambda

    def forward(self, a, b, c, d):
        cos_loss = torch.nn.CosineSimilarity()
        loss1 = 0
        loss2 = 0
        loss3 = 0
        for item in range(len(a)):
            loss1 += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1))) 
            loss2 += torch.mean(1-cos_loss(b[item].view(b[item].shape[0],-1),c[item].view(c[item].shape[0],-1))) * self.DLlambda
            loss3 += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),d[item].view(d[item].shape[0],-1))) * self.DLlambda
        loss = loss1+loss2+loss3
        return loss

class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x, device, reduction ='mean'):
        if reduction == 'none':
            return (x - x_hat)**2
        elif reduction == 'mean':
            return torch.mean((x - x_hat)**2)
        elif reduction == 'batch':
            return torch.mean((x - x_hat)**2, dim=(1,2,3))

class SSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x, device, reduction ='mean'):
        if reduction == 'none':
            return (x - x_hat)**2
        elif reduction == 'mean':
            return torch.sum((x - x_hat)**2)
        elif reduction == 'batch':
            return torch.sum((x - x_hat)**2, dim=(1,2,3))

class L1norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x, device, reduction ='mean'):
        if reduction == 'none':
            return torch.abs(x - x_hat)
        elif reduction == 'mean':
            return torch.mean(torch.abs(x - x_hat))
        elif reduction == 'batch':
            return torch.mean(torch.abs(x - x_hat), dim=(1,2,3))
        
class SSIM_loss(nn.Module):
    def __init__(self):

        super().__init__()
        self.kernel_size = 5
        self.sigma = 1.5
        self.data_range = (-1., 1.)

    # def normalize(self, x):
    #     min = x.min(dim=0, keepdim=True)[0]
    #     max = x.max(dim=0, keepdim=True)[0]
    #     x_norm = (x - min) / (max - min + 1e-8)
    #     return x_norm
        
    def forward(self, x, x_hat, device, reduction = 'mean'):
        if reduction == 'none':
            return (x - x_hat)**2
        elif reduction == 'mean':
            ssim = SSIM(kernel_size=self.kernel_size,
                    sigma = self.sigma,
                    data_range = self.data_range,
                    reduction = 'elementwise_mean').to(device)
            return 1- ssim(x_hat, x)
        elif reduction == 'batch':
            ssim = SSIM(kernel_size=self.kernel_size,
                    sigma = self.sigma,
                    data_range = self.data_range,
                    reduction = 'none').to(device)
            return 1- ssim(x_hat, x)

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