# Test of periodic_noise function
import numpy as np
from periodic_noise import periodic_noise
from matplotlib import pyplot as plt
from skimage.morphology import square, dilation
import skimage
import cv2

img = cv2.imread('./imgs/Apollo17boulder.tif', cv2.IMREAD_GRAYSCALE)

fig, axs = plt.subplots(ncols=3, figsize=(12,6))
ax = axs.ravel()
[axi.set_axis_off() for axi in ax.ravel()]

ax[0].imshow(img, cmap='gray', vmin = 1.0, vmax = 255)
ax[0].set_title('Original')

C = np.array([[24, 44], [23, 44], [23, 43],[24, 43]]) 

r, R = periodic_noise(img.shape, C)
g = img + r
G = np.abs(np.fft.fftshift(np.fft.fft2(g)))
Gd =dilation(np.log(1+G),square(3))

ax[1].imshow(Gd, cmap='gray')
ax[1].set_title('Frequency noise R')

ax[2].imshow(g, cmap='gray', vmin = 1.0, vmax = 255)
ax[2].set_title('spatial noise r')