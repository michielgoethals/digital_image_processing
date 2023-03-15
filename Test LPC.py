##test of LPC

from Lp_Hp_Filters import lpcfilter
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import (fft2, ifft2, fftshift)
import cv2

letter = cv2.imread('.\imgs\LetterA.tif', cv2.IMREAD_GRAYSCALE)
F = fftshift(fft2(letter))

ideal_5 = lpcfilter(letter.shape, ftype='ideal', D0 = 5)
ideal_5 = np.real(ifft2(fftshift(F*ideal_5), letter.shape))

ideal_15 = lpcfilter(letter.shape, ftype='ideal', D0 = 15)
ideal_15 = np.real(ifft2(fftshift(F*ideal_15), letter.shape))

ideal_30 = lpcfilter(letter.shape, ftype='ideal', D0 = 30)
ideal_30 = np.real(ifft2(fftshift(F*ideal_30), letter.shape))

ideal_80 = lpcfilter(letter.shape, ftype='ideal', D0 = 80)
ideal_80 = np.real(ifft2(fftshift(F*ideal_80), letter.shape))

ideal_230 = lpcfilter(letter.shape, ftype='ideal', D0 = 230)
ideal_230 = np.real(ifft2(fftshift(F*ideal_230), letter.shape))

fig, axes = plt.subplots(ncols=6, figsize=(20, 5))
ax = axes.ravel()
[axi.set_axis_off() for axi in ax.ravel()]
ax[0].imshow(letter, cmap='gray', vmin = 1.0, vmax = 255)
ax[0].set_title('Original')
ax[1].imshow(ideal_5, cmap='gray', vmin = 1.0, vmax = 255)
ax[1].set_title('Ideal LPC D0 = 5')
ax[2].imshow(ideal_15, cmap='gray', vmin = 1.0, vmax = 255)
ax[2].set_title('Ideal LPC D0 = 30')
ax[3].imshow(ideal_30, cmap='gray', vmin = 1.0, vmax = 255)
ax[3].set_title('Ideal LPC D0 = 80')
ax[4].imshow(ideal_80, cmap='gray', vmin = 1.0, vmax = 255)
ax[4].set_title('Ideal LPC D0 = 80')
ax[5].imshow(ideal_230, cmap='gray', vmin = 1.0, vmax = 255)
ax[5].set_title('Ideal LPC D0 = 230')
plt.show()

gaussian_5 = lpcfilter(letter.shape, ftype='gaussian', D0 = 5)
gaussian_5 = np.real(ifft2(fftshift(F*gaussian_5), letter.shape))

gaussian_15 = lpcfilter(letter.shape, ftype='gaussian', D0 = 15)
gaussian_15 = np.real(ifft2(fftshift(F*gaussian_15), letter.shape))

gaussian_30 = lpcfilter(letter.shape, ftype='gaussian', D0 = 30)
gaussian_30 = np.real(ifft2(fftshift(F*gaussian_30), letter.shape))

gaussian_85 = lpcfilter(letter.shape, ftype='gaussian', D0 = 85)
gaussian_85 = np.real(ifft2(fftshift(F*gaussian_85), letter.shape))

gaussian_230 = lpcfilter(letter.shape, ftype='gaussian', D0 = 230)
gaussian_230 = np.real(ifft2(fftshift(F*gaussian_230), letter.shape))

fig, axes = plt.subplots(ncols=6, figsize=(20, 5))
ax = axes.ravel()
[axi.set_axis_off() for axi in ax.ravel()]
ax[0].imshow(letter, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(gaussian_5, cmap='gray', vmin = 1.0, vmax = 255)
ax[1].set_title('Gaussian LPC D0 = 5')
ax[2].imshow(gaussian_15, cmap='gray', vmin = 1.0, vmax = 255)
ax[2].set_title('Gaussian LPC D0 = 30')
ax[3].imshow(gaussian_30, cmap='gray', vmin = 1.0, vmax = 255)
ax[3].set_title('Gaussian LPC D0 = 80')
ax[4].imshow(gaussian_85, cmap='gray', vmin = 1.0, vmax = 255)
ax[4].set_title('Gaussian LPC D0 = 80')
ax[5].imshow(gaussian_230, cmap='gray', vmin = 1.0, vmax = 255)
ax[5].set_title('Gaussian LPC D0 = 230')
plt.show()

butterworth_5 = lpcfilter(letter.shape, ftype='butterworth', n=2, D0 = 5)
butterworth_5 = np.real(ifft2(fftshift(F*butterworth_5), letter.shape))

butterworth_15 = lpcfilter(letter.shape, ftype='butterworth', n=2, D0 = 15)
butterworth_15 = np.real(ifft2(fftshift(F*butterworth_15), letter.shape))

butterworth_30 = lpcfilter(letter.shape, ftype='butterworth', n=2, D0 = 30)
butterworth_30 = np.real(ifft2(fftshift(F*butterworth_30), letter.shape))

butterworth_80 = lpcfilter(letter.shape, ftype='butterworth', n=2, D0 = 80)
butterworth_80 = np.real(ifft2(fftshift(F*butterworth_80), letter.shape))

butterworth_230 = lpcfilter(letter.shape, ftype='butterworth', n=2, D0 = 230)
butterworth_230 = np.real(ifft2(fftshift(F*butterworth_230), letter.shape))

fig, axes = plt.subplots(ncols=6, figsize=(20, 5))
ax = axes.ravel()
plt.axis('off')
[axi.set_axis_off() for axi in ax.ravel()]
ax[0].imshow(letter, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(butterworth_5, cmap='gray', vmin = 1.0, vmax = 255)
ax[1].set_title('Butterworth LPC D0 = 5')
ax[2].imshow(butterworth_15, cmap='gray', vmin = 1.0, vmax = 255)
ax[2].set_title('Butterworth LPC D0 = 30')
ax[3].imshow(butterworth_30, cmap='gray', vmin = 1.0, vmax = 255)
ax[3].set_title('Butterworth LPC D0 = 80')
ax[4].imshow(butterworth_80, cmap='gray', vmin = 1.0, vmax = 255)
ax[4].set_title('Butterworth LPC D0 = 80')
ax[5].imshow(butterworth_230, cmap='gray', vmin = 1.0, vmax = 255)
ax[5].set_title('Butterworth LPC D0 = 230')
plt.show()