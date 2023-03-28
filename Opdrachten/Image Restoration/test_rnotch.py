# Test of rnotch_filter

import numpy as np
import matplotlib.pyplot as plt
from rnotch_filter import rnotch_filter
import skimage
import imageio

img = imageio.imread('./imgs/saturn_rings.tif') 
img = skimage.util.img_as_float(img) 

F = np.fft.fftshift(np.fft.fft2(img))

H = rnotch_filter(img.shape, D0=10, ftype='ideal', W=10, angle=90) 

G = F*H
g = np.real(np.fft.ifft2 (np.fft.fftshift(G))) 

fig, (ax1, ax2,ax3,ax4) = plt.subplots (nrows=4, ncols=1, figsize=(3,12), sharex = True) 

ax1.imshow (img) ;ax1.axis('off'); 
ax1.set_title('Noisy image') 
ax2.imshow (np.log(1+np.abs(F))) ;ax2.axis('off'); 
ax3.set_title('Fourier spectrum') 
ax3.imshow(np. log (1+np.abs (G))); ax3.axis('off'); 
ax3.set_title('Filtered Fourier spectrum') 
ax4.imshow (g) ;ax4.axis ('off'); 
ax4.set_title('Filtered image')