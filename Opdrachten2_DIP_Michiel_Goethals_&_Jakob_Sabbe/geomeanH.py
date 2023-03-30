from dftfilt import dftfilt
import numpy as np
import imageio
import skimage
import matplotlib.pyplot as plt
from Lp_Hp_Filters import lpcfilter
from skimage.util.noise import random_noise
from scipy import ndimage as ndi

def get_geomean_H(H,alpha=1,beta=1,Sn=None,Sf=None):
    """ Construct the geometric mean filter from frequencty filter H"""
    
    if (Sn is None and Sf is None):
        Sn = 1
        Sf = 1
       
    Hk = (((1/H)*np.abs(H*H)/np.abs(H*H))**alpha) * (((1/H)*np.abs(H*H)/(np.abs(H*H) + beta*(Sn/Sf)))**(1-alpha))
    
    return Hk

if __name__ == "__main__":
    img = imageio.imread('./imgs/obelix.tif')
    img = skimage.util.img_as_float(img)
    
    H = lpcfilter(img.shape, ftype='gaussian', D0=30)
    
    g = dftfilt(img,H)
    
    gn = random_noise(g, mode='s&p', amount=0.001)
    fn_ei = dftfilt(gn, 1/H)
    
    gnm = ndi.median_filter(gn,3)
    fnm_ei = dftfilt(gnm, 1/H)
    
    H_geomean = get_geomean_H(H,alpha=0,beta=0.01)
    
    fn_ew2 = dftfilt(gn,H_geomean)
    fnm_ew2 = dftfilt(gnm,H_geomean)

    plt.figure(figsize=(8,3))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,3))
    axes[0].imshow(img, cmap='gray');
    axes[0].axis('off');
    axes[0].set_title('Original image')
    axes[1].imshow(fn_ew2, cmap='gray');
    axes[1].axis('off');
    axes[1].set_title('Restored noisy image')
    axes[2].imshow(fnm_ew2, cmap='gray');
    axes[2].axis('off');
    axes[2].set_title('Restored denoised image')