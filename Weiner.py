from dftfilt import dftfilt
import numpy as np
import imageio
import skimage
import matplotlib.pyplot as plt
from Lp_Hp_Filters import lpcfilter
from skimage.util.noise import random_noise
from scipy import ndimage as ndi

def get_weiner_H(H, k=1, Sn=None, Sf=None):
    """Construct the Weiner filter from frequency filter H. """
    
    Hw = np.zeros_like(H)
    
    if(Sn is not None and Sf is not None):
        k = Sn//Sf
     
    Hw = 1/H *np.abs(H*H)/(np.abs(H*H)+k)
    
    return Hw
    

if __name__ == "__main__":
    img = imageio.imread('./imgs/obelix.tif')
    img = skimage.util.img_as_float(img)
    
    H = lpcfilter(img.shape, ftype='gaussian', D0=30)
    
    g = dftfilt(img,H)
    
    gn = random_noise(g, mode='s&p', amount=0.001)
    fn_ei = dftfilt(gn, 1/H)
    
    gnm = ndi.median_filter(gn,3)
    fnm_ei = dftfilt(gnm, 1/H)
    
    Hw = get_weiner_H(H, k=0.01)
    
    fn_ew = dftfilt(gn, Hw)
    fnm_ew = dftfilt(gnm, Hw)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
    axes[0].imshow(fn_ew, cmap='gray');
    axes[0].axis('off');
    axes[0].set_title('Restored noisy image')
    axes[1].imshow(fnm_ew, cmap='gray');
    axes[1].axis('off');
    axes[1].set_title('Restored denoised image')
    
    
    