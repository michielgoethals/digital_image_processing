# Test of periodic_noise function

import cv2
import numpy as np
from periodic_noise import periodic_noise
from matplotlib import pyplot as plt
from skimage.morphology import square, dilation


img = cv2.imread('./imgs/Apollo17boulder.tif', cv2.IMREAD_GRAYSCALE)

C = np.array([[24, 44]]) 

r, R = periodic_noise(img.shape, C)

g = img+r
G = np.fft.fftshift(np.fft.fft2(g))
Gd = dilation(np.log(1+np.abs(G)), square(3))

plt.subplot(131);plt.axis('off');plt.imshow(img, cmap='gray')
plt.subplot(132);plt.axis('off');plt.imshow(g, cmap='gray')
plt.subplot(133);plt.axis('off');plt.imshow(Gd, cmap='gray')
