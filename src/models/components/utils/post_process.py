import numpy as np
from skimage.metrics import structural_similarity
import torch
from torchvision.transforms.functional import rgb_to_grayscale
from skimage.filters import sobel
import skimage.morphology as morphology

def to_gray_0_1(x):
    """
    Convert the first 3 channels (RGB) of the input tensor to grayscale, 
    concatenate the result with the remaining channels, 
    and normalize the tensor to the range [0, 1].

    Args:
        x (torch.Tensor): Input tensor with at least 4 channels, where the first 3 channels represent 
        RGB values and the fourth channel represents height or another feature.

    Returns:
        torch.Tensor: Tensor with the first 3 channels converted to grayscale, 
        concatenated with the remaining channels, and normalized to the range [0, 1].
    """
    # Convert the first 3 channels (RGB) to grayscale
    x_gray = rgb_to_grayscale(x[:, :3])
    
    # Concatenate the grayscale result with the remaining channels
    x = torch.cat((x_gray, x[:, 3:]), dim=1)
    
    # Normalize the tensor to the range [0, 1]
    x = (x + 1) / 2
    
    return x

def ssim_for_batch(batch, r_batch, win_size=5):
    """
    Calculate the Structural Similarity Index (SSIM) for a batch of images.

    Args:
        batch (torch.Tensor): Batch of images to be compared.
        r_batch (torch.Tensor): Reference batch of images.
        win_size (int, optional): The size of the window to use for SSIM calculation. Default is 5.

    Returns:
        ssim_batch (numpy.ndarray): Array containing the SSIM values for each image in the batch.
        ssim_batch_img (numpy.ndarray): Array containing the SSIM images for each image in the batch.
    """
    # Convert the batches from PyTorch tensors to NumPy arrays
    batch   = batch.cpu().numpy()
    r_batch = r_batch.cpu().numpy()
    
    # Get the batch size
    bs = batch.shape[0]
    
    # Initialize arrays to store SSIM values and SSIM images
    ssim_batch      = np.zeros((batch.shape[0], batch.shape[1]))
    ssim_batch_img  = np.zeros_like(batch)
    
    # Loop through each image in the batch
    for i in range(bs):
        for j in range(batch.shape[1]):
            # Calculate SSIM and SSIM image for the current image
            ssim, img_ssim = structural_similarity(batch[i, j], 
                                                   r_batch[i, j],
                                                   win_size=win_size,
                                                   data_range=1,
                                                   full=True)
            # Invert the SSIM image
            ssim_batch_img[i, j] = img_ssim * -1
            # Count the number of positive values in the inverted SSIM image
            ssim_batch[i, j] = np.sum(ssim_batch_img[i, j] > 0)
    
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

def individual_post_processing(x0, x1, idx):
    """
    Given the original sample x0 and its reconstructions x1, this functions 
    returns a plot with the intermediate results of post processing.

    Args:
        x0 (2D tensor) : input sample (Bx2xHxW)
        x1 (2D tensor) : reconstruction of x0 (Bx2xHxW)
        idx (int)      : sample index

    Returns:
        ssim_img (2D tensor) : reconstruction of x0 (2xHxW)
        filt1 (2D tensor) : filtered ssim map thresholding (2xHxW)
        filt2 (2D tensor) : filtered ssim map thresholding + erosion (2xHxW)
        ano_map (2D tensor) : filtered anomaly map (HxW)
    """
    _, ssim_img             = ssim_for_batch(x0, x1)

    # Create empty tensor for filtered ssim and anomaly maps
    ssim_filt1  = np.zeros_like(ssim_img[0])
    ssim_filt2  = np.zeros_like(ssim_img[0])

    # Loop over images in batch and both channels. Necessary since
    # skimage has no batch processing
    for i in range(ssim_img.shape[1]):
        
        # Thresholding
        ssim_filt1[i] = (ssim_img[idx,i] > np.percentile(ssim_img[idx,i], q=94)).astype(int)
        
        # Morphology filters
        ssim_filt2[i] = morphology.binary_erosion(ssim_filt1[i])

    # Boolean masks: if pixel is present in ssim height, ssim rgb
    # and sobel filter, it is accounted as crack pixel  
    ano_map = (
                    (ssim_filt2[0]   == 1) & 
                    (ssim_filt2[1]   == 1) 
                    ).astype(int)
    
    return ssim_img[idx], ssim_filt1, ssim_filt2, ano_map


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