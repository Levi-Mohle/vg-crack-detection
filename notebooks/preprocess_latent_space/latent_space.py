import torch

def encode_decode(vae, rgb, height, device="cpu"):
    """
    Encodes and subsequently decodes rgb and height images with given pre-trained vae

    Args:
        vae (AutoEncoderKL): pre-trained vae
        rgb (Tensor) : Tensor containing rgb images [N,3,h,w]
        height (Tensor): Tensor containing height images [N,1,h,w]

    Returns:
        recon_rgb (Tensor) : Tensor containing recontructed rgb images [N,3,h,w]
        recon_height (Tensor): Tensor containing reconstructed height images [N,1,h,w]
    """
    with torch.no_grad():
        # Encode
        enc_rgb, enc_height = encode_2ch(vae, rgb, height, device)  
        # Decode
        recon_rgb, recon_height = decode_2ch(vae, enc_rgb, enc_height, device)
    
    return recon_rgb.cpu(), recon_height.cpu()

def encode_(vae,x,device):
    """
    Basis function for encoding data with vae

    Args:
        vae (AutoEncoderKL): pre-trained vae
        x (Tensor) : Tensor containing 3 channel images [N,3,h,w]

    Returns:
        z (Tensor) : Tensor containing encoded rgb images [N,4,h/8,w/8]
    """
    z = vae.encode(x.to(device)).latent_dist.sample().mul_(0.18215)
    return z

def encode_2ch(vae, rgb, height, device="cpu"):
    """
    Encodes rgb and height images with given pre-trained vae

    Args:
        vae (AutoEncoderKL): pre-trained vae
        rgb (Tensor) : Tensor containing rgb images [N,3,h,w]
        height (Tensor): Tensor containing height images [N,1,h,w]

    Returns:
        enc_rgb (Tensor) : Tensor containing encoded rgb images [N,4,h/8,w/8]
        enc_height (Tensor): Tensor containing encoded height images [N,4,h/8,w/8]
    """
    # Duplicate height channel to fit the vae
    height = torch.cat((height,height,height), dim=1)
    
    vae.to(device)

    # Encode
    with torch.no_grad():
        enc_rgb     = encode_(vae, rgb, device)
        enc_height  = encode_(vae, height, device)

    return enc_rgb, enc_height

def decode_(vae,z,device):
    """
    Basis function for decoding data with vae

    Args:
        vae (AutoEncoderKL): pre-trained vae
        z (Tensor) : Tensor containing encoded rgb images [N,4,h/8,w/8]

    Returns:
        x (Tensor) : Tensor containing 3 channel images [N,3,h,w]
    """
    xhat = vae.encode(z.to(device)/0.18215).sample
    return xhat

def decode_2ch(vae, enc_rgb, enc_height, device="cpu"):
    """
    Decodes rgb and height images with given pre-trained vae

    Args:
        vae (AutoEncoderKL): pre-trained vae
        enc_rgb (Tensor) : Tensor containing encoded rgb images [N,4,h/8,w/8]
        enc_height (Tensor): Tensor containing encoded height images [N,4,h/8,w/8]

    Returns:
        rgb (Tensor) : Tensor containing rgb images [N,3,h,w]
        height (Tensor): Tensor containing height images [N,1,h,w]
    """
    vae.to(device)
    
    # Decode
    recon_rgb      = decode_(enc_rgb)
    recon_height   = decode_(enc_height)[:,0].unsqueeze(1)

    return recon_rgb, recon_height