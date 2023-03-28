import imageio
import numpy as np
import skimage
import matplotlib.pyplot as plt
from dftfilt import dftfilt
from Lp_Hp_Filters import lpcfilter
from skimage.util.noise import random_noise
from scipy import ndimage as ndi


def get_inv_H(H, cutoff=None, eps=1e-16):
    """
    Construct the inverse filter from frequency filter H.
    cutoff: if None return 1/H (direct inverse filter, values in H smaller 
            than eps are first clipped to eps)
        if not None: return radially limited inverse filter
                with values of frequencies larger
                than cutoff set to eps.
    """
    
    Hi = np.zeros_like(H)

    Hc = np.clip(H, eps, None)
    Hi = 1/Hc
        
    if cutoff is not None:   
        r, c = H.shape
        R, C = np.ogrid[:r,:c]
        D = np.sqrt((R-r//2)**2 + (C-c//2)**2)
        Hi[D > cutoff] = eps
        
    return Hi


if __name__ is "__main__":
    img = imageio.imread('./imgs/obelix.tif')
    img = skimage.util.img_as_float(img)
    
    H = lpcfilter(img.shape, ftype='gaussian', D0=20)
    
    g = dftfilt(img,H)

    cutoffs= [30, 70, 100]
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8,3))
    
    axes[0].imshow(g, cmap='gray')
    axes[0].axis('off');
    axes[0].set_title('Filtered')
    
    for i,x in enumerate(cutoffs):
        Hi = get_inv_H(H, cutoff=x)
        f_est = dftfilt(g, Hi)
        axes[i+1].imshow(f_est, cmap='gray');
        axes[i+1].axis('off');
        axes[i+1].set_title('cutoff = {:1.0f}'.format(x))
    
    axes[3].imshow(g, cmap='gray')