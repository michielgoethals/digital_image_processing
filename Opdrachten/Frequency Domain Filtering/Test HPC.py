##test of HPC

from Lp_Hp_Filters import hpcfilter
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import (fft2, ifft2, fftshift)
import cv2

letter = cv2.imread('.\imgs\LetterA.tif', cv2.IMREAD_GRAYSCALE)
F = fftshift(fft2(letter))

ideal_15 = hpcfilter(letter.shape, ftype='ideal', D0 = 15)
ideal_15 = np.real(ifft2(fftshift(F*ideal_15), letter.shape))

ideal_30 = hpcfilter(letter.shape, ftype='ideal', D0 = 30)
ideal_30 = np.real(ifft2(fftshift(F*ideal_30), letter.shape))

ideal_80 = hpcfilter(letter.shape, ftype='ideal', D0 = 80)
ideal_80 = np.real(ifft2(fftshift(F*ideal_80), letter.shape))

fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
ax = axes.ravel()
[axi.set_axis_off() for axi in ax.ravel()]
ax[0].imshow(letter, cmap='gray', vmin = 1.0, vmax = 255)
ax[0].set_title('Original')
ax[1].imshow(ideal_15, cmap='gray', vmin = 1.0, vmax = 255)
ax[1].set_title('Ideal HPC D0 = 15')
ax[2].imshow(ideal_30, cmap='gray', vmin = 1.0, vmax = 255)
ax[2].set_title('Ideal HPC D0 = 30')
ax[3].imshow(ideal_80, cmap='gray', vmin = 1.0, vmax = 255)
ax[3].set_title('Ideal HPC D0 = 80')
plt.show()

gaussian_15 = hpcfilter(letter.shape, ftype='gaussian', D0 = 15)
gaussian_15 = np.real(ifft2(fftshift(F*gaussian_15), letter.shape))

gaussian_30 = hpcfilter(letter.shape, ftype='gaussian', D0 = 30)
gaussian_30 = np.real(ifft2(fftshift(F*gaussian_30), letter.shape))

gaussian_80 = hpcfilter(letter.shape, ftype='gaussian', D0 = 85)
gaussian_80 = np.real(ifft2(fftshift(F*gaussian_80), letter.shape))


fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
ax = axes.ravel()
[axi.set_axis_off() for axi in ax.ravel()]
ax[0].imshow(letter, cmap='gray', vmin = 0, vmax = 255)
ax[0].set_title('Original')
ax[1].imshow(gaussian_15, cmap='gray', vmin = 1.0, vmax = 255)
ax[1].set_title('Gaussian HPC D0 = 15')
ax[2].imshow(gaussian_30, cmap='gray', vmin = 1.0, vmax = 255)
ax[2].set_title('Gaussian HPC D0 = 30')
ax[3].imshow(gaussian_80, cmap='gray', vmin = 1.0, vmax = 255)
ax[3].set_title('Gaussian HPC D0 = 80')
plt.show()

butterworth_15 = hpcfilter(letter.shape, ftype='butterworth', D0 = 15)
butterworth_15 = np.real(ifft2(fftshift(F*butterworth_15), letter.shape))

butterworth_30 = hpcfilter(letter.shape, ftype='butterworth', D0 = 30)
butterworth_30 = np.real(ifft2(fftshift(F*butterworth_30), letter.shape))

butterworth_80 = hpcfilter(letter.shape, ftype='butterworth', D0 = 80)
butterworth_80 = np.real(ifft2(fftshift(F*butterworth_80), letter.shape))

fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
ax = axes.ravel()
[axi.set_axis_off() for axi in ax.ravel()]
ax[0].imshow(letter, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(butterworth_15, cmap='gray', vmin = 1.0, vmax = 255)
ax[1].set_title('Butterworth HPC D0 = 15')
ax[2].imshow(butterworth_30, cmap='gray', vmin = 1.0, vmax = 255)
ax[2].set_title('Butterworth HPC D0 = 30')
ax[3].imshow(butterworth_80, cmap='gray', vmin = 1.0, vmax = 255)
ax[3].set_title('Butterworth HPC D0 = 80')
plt.show()