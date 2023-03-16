# Test of periodic_noise function

import cv2
import numpy as np
from periodic_noise import periodic_noise
from matplotlib import pyplot as plt


img = cv2.imread('./imgs/Apollo17boulder.tif', cv2.IMREAD_GRAYSCALE)

fig, axs = plt.subplots(ncols=3, figsize=(12,6))
ax = axs.ravel()
[axi.set_axis_off() for axi in ax.ravel()]

ax[0].imshow(img, cmap='gray', vmin = 1.0, vmax = 255)
ax[0].set_title('Original')

C = np.array([[24, 44]]) 

r, R = periodic_noise(img.shape, C)

ax[1].imshow(R, cmap='gray', vmin = 1.0, vmax = 255)
ax[1].set_title('Frequency noise R')

ax[2].imshow(r, cmap='gray', vmin = 1.0, vmax = 255)
ax[2].set_title('spatial noise r')

