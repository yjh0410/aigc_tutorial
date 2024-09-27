import numpy as np
import torch
import torch.nn.functional as F


def batch_calculate_psnr(images_true, images_test, d_max=None):
    """
        Calculate peak signal noise ratio (PSNR).
        Inputs:
            images_true: (torch.FloatTensor) [B, C, H, W]
            images_test: (torch.FloatTensor) [B, C, H, W]
    """
    assert images_true.shape == images_test.shape, "Please make sure that the shapes of two input images is equal to each other."
    # Compute MSE
    mse = F.mse_loss(images_true, images_test, reduction='none')  # [B, C, H, W]
    mse = torch.mean(mse, dim=[1, 2, 3]) # [B,]

    d_max = 1.0 if d_max is None else d_max
    psnr = 20 * torch.log10(d_max / torch.sqrt(mse))
    
    return psnr.mean()

def batch_calculate_ssim(images_true, images_test, window_size=None):
    """
        Calculate structural similarity (SSIM).
        Inputs:
            images_true: (torch.FloatTensor) [B, C, H, W]
            images_test: (torch.FloatTensor) [B, C, H, W]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    sigma = 1.5
    channel = images_test.shape[1]

    if window_size is None:
        window_size = 11
        # Gaussian
        _1d_window = torch.as_tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        _1d_window = _1d_window / _1d_window.sum()
        _1d_window = _1d_window.unsqueeze(1)
        # [ws, ws]
        _2d_window = torch.mm(_1d_window, _1d_window.t()).float()
        # [ws, ws] -> [c, 1, ws, ws]
        window = _2d_window[None, None].repeat(channel, 1, 1, 1)
        window = window.to(images_test.device)

    mu1 = F.conv2d(images_true, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(images_test, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(images_true * images_true, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(images_test * images_test, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(images_true * images_test, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = torch.mean(ssim, dim=[1, 2, 3])

    return ssim.mean()
