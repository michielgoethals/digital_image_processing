import imageio
import numpy as np
import skimage
from Lp_Hp_Filters import lpcfilter
import matplotlib.pyplot as plt

def dftfilt(fg, H, pad=False):
    """
    Filter the image f in the frequency domain with the filter H (G = F*H) 
    and return the filtered image g. 
    H: filter in frequency domain (centered at image center) 
    pad: using padding before filtering 
    Note: if pad == True: H.shape should be 2*img.shape!
    """
    
    M, N = fg.shape
    
    pad_size = ((M//2,M//2),(N//2,N//2))
    
    if pad:
        fg = np.pad(fg, pad_size, mode='constant')
        
    # Compute the 2D Fourier transforms of the input image and the filter
    F = np.fft.fft2(fg)
    H = np.fft.fftshift(H)
    G = F * H
    
    if pad:
        g = np.fft.ifft2(G)[M//2:3*M//2,N//2:3*N//2].real
    else:
        g = np.fft.ifft2(G).real
    
    return g
# Test function

if __name__ == "__main__":

    img = imageio.imread('./imgs/obelix.tif')
    img = skimage.util.img_as_float(img)
    r,c = img.shape
    
    H = lpcfilter((r,c), ftype='gaussian', D0=30)
    g = dftfilt(img, H)
    
    Hp = lpcfilter((2*r,2*c), ftype='gaussian', D0=30*2)
    gp = dftfilt(img, Hp, pad=True)
    
    f_ei = dftfilt(g, 1/H)
    fp_ei = dftfilt(gp, 1/Hp, pad=True)
    
    fig, axes = plt.subplots(ncols=5, figsize=(19, 4))
    ax = axes.ravel()
    [axi.set_axis_off() for axi in ax.ravel()]
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(g, cmap='gray')
    ax[1].set_title('Filtered (no padding)')
    ax[2].imshow(gp, cmap='gray')
    ax[2].set_title('Filtered (with padding)')
    ax[3].imshow(f_ei, cmap='gray')
    ax[3].set_title('Deconvoluted (no padding)')
    ax[4].imshow(fp_ei, cmap='gray')
    ax[4].set_title('Deconvoluted (with padding)')
    