import numpy as np
from skimage.metrics import structural_similarity
import torch
from torchvision.transforms.functional import rgb_to_grayscale
from skimage.filters import sobel
import skimage.morphology as morphology

def to_gray_0_1(x):
     # Convert first 3 channels (rgb) to gray-scale
     x_gray = rgb_to_grayscale(x[:,:3])
     # Concatentate result with height channel
     x = torch.cat((x_gray, x[:,3:]), dim=1)
     # Normalize back to [0,1]
     x = (x+1)/2
     return x

def ssim_for_batch(batch, r_batch, win_size=5):
    batch   = batch.cpu().numpy()
    r_batch = r_batch.cpu().numpy()
    bs = batch.shape[0]
    
    ssim_batch     = np.zeros((batch.shape[0],batch.shape[1]))
    ssim_batch_img = np.zeros_like(batch)
    for i in range(bs):
        for j in range(batch.shape[1]):
            ssim,  img_ssim = structural_similarity(batch[i,j], 
                                r_batch[i,j],
                                win_size=win_size,
                                data_range=1,
                                full=True)
            ssim_batch_img[i, j] = img_ssim * -1
            ssim_batch[i, j]     = np.sum(ssim_batch_img[i, j] > 0)
    
    return ssim_batch, ssim_batch_img

def post_processing(ssim_img):
    """
    Given the input sample x0 and anomaly maps produced with SSIM,
    this function filters the anomaly maps of noise and non-crack 
    related artifacts. It derives an OOD-score from the filtered map.

    Args:
        ssim_img (2D tensor) : reconstruction of x0 (Bx2xHxW)
        
    Returns:
        ano_maps (2D tensor) : filtered anomaly map (Bx1xHxW)

    """
    # Create empty tensor for filtered ssim and anomaly maps
    ssim_filt = np.zeros_like(ssim_img)
    ano_maps  = np.zeros((ssim_img.shape[0],ssim_img.shape[2],ssim_img.shape[3]))
    # sobel_filt  = np.zeros((ssim_img.shape[0],ssim_img.shape[2],ssim_img.shape[3]))

    # Loop over images in batch and both channels. Necessary since
    # skimage has no batch processing
    for idx in range(ssim_img.shape[0]):

        for i in range(ssim_img.shape[1]):
            
            # Thresholding
            ssim_filt[idx,i] = (ssim_img[idx,i] > np.percentile(ssim_img[idx,i], q=94)).astype(int)
            
            # Morphology filters
            ssim_filt[idx,i] = morphology.binary_erosion(ssim_filt[idx,i])


        # Boolean masks: if pixel is present in ssim height, ssim rgb
        # and sobel filter, it is accounted as crack pixel  
        ano_maps[idx] = (
                        (ssim_filt[idx,0]   == 1) & 
                        (ssim_filt[idx,1]   == 1) 
                        ).astype(int)
                
    return ano_maps

def get_OOD_score(x0, x1):
    """
    Given the original sample x0 and its reconstructions x1 and x2, 
    this function returns the filtered anomaly map and OOD-score to be
    used in classification. If comparison is made between x0 and x1 or x2,
    provide x1 = x0.

    Args:
        x0 (2D tensor) : input sample (Bx2xHxW)
        x1 (2D tensor) : reconstruction of x0 (Bx2xHxW)
        

    Returns:
        ood_score (1D tensor) : out-of-distribution scores (Bx1)
    
    """
    # Obtain SSIM between x1 and x2
    _, ssim_img             = ssim_for_batch(x0, x1)

    # Calculate anomaly maps and OOD-score
    ano_maps                = post_processing(ssim_img)

    # Calculate OOD-score, based on total number of crack pixels
    ood_score = np.sum(ano_maps, axis=(1,2))
    
    return ood_score