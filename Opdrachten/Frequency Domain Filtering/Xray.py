#dia 51: High frequency emphasis filtering and histogram equalization for image enhancement

import cv2
from numpy.fft import (fft2, ifft2, fftshift)
import matplotlib.pyplot as plt
import numpy as np
from Lp_Hp_Filters import hpcfilter

img = cv2.imread('./imgs/ChestXray.tif', cv2.IMREAD_GRAYSCALE)
F = fftshift(fft2(img))

fig, axs = plt.subplots(2, 2, figsize=(12,8))
ax = axs.ravel()
[axi.set_axis_off() for axi in ax.ravel()]


axs[0,0].imshow(img, cmap='gray', vmin = 1.0, vmax = 255)
axs[0,0].set_title("Original image")

H = hpcfilter(img.shape, ftype = 'gaussian', D0 = 8)

highpass = np.real(ifft2(fftshift(H*F), img.shape))

axs[0,1].imshow(highpass, cmap='gray', vmin = 1.0, vmax = 255)
axs[0,1].set_title("Highpass filtering")

a = 1
b = 3

high_freq_emph = np.real(ifft2(fftshift((a+ b*H)*F), img.shape))

axs[1,0].imshow(high_freq_emph, cmap='gray')
axs[1,0].set_title("High frequency emphasis result")


#Normalizing the image in order to equlize it afterwards
high_freq_emph = cv2.normalize(high_freq_emph, None, 0, 255, cv2.NORM_MINMAX,  dtype=cv2.CV_8UC1)
hist_eq = cv2.equalizeHist(high_freq_emph)

axs[1,1].imshow(hist_eq, cmap='gray', vmin = 1.0, vmax = 255)
axs[1,1].set_title('After histogram equalisation')

fig.show()