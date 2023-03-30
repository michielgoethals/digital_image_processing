# Test of cnotch

import numpy as np
from cnotch_filter import cnotch_filter
import matplotlib.pyplot as plt
from skimage.morphology import square, dilation
import cv2

img = cv2.imread('./imgs/Apollo17boulder.tif', cv2.IMREAD_GRAYSCALE)/255

M, N = img.shape

thetas = np.array([0, 60, 120])
D = 30*np.sqrt(2)
u = (D*np.cos(thetas*np.pi/180) + M//2).astype(int)
v = (D*np.sin(thetas*np.pi/180) + N//2).astype(int)

A = np.ones((thetas.size,))
R = np.zeros((M,N), dtype = complex)
R[u,v] = 1j * (A/2) * M * N * np.exp(0)

r = np.real(np.fft.ifft2(np.fft.ifftshift(R)))

g = img + r/3

G = np.fft.fftshift(np.fft.fft2(g))

Gd = dilation(np.log(1+np.abs(G)),square(3))
plt.figure();plt.axis('off');plt.imshow(Gd)

xy = np.array(plt.ginput(-1, show_clicks=True))
rc = xy[:,::-1]


center = np.array(img.shape)//2
rc = rc - center

H = cnotch_filter(img.shape,rc,ftype='ideal',reject = True,D0=3)

G2 = G*H
g2 = np.real(np.fft.ifft2(np.fft.ifftshift(G2)))

plt.subplot(411);plt.axis('off');plt.imshow(g, cmap = 'gray')
plt.subplot(412);plt.axis('off');plt.imshow(Gd, cmap = 'gray')
plt.subplot(413);plt.axis('off');plt.imshow(H, cmap = 'gray')
plt.subplot(414);plt.axis('off');plt.imshow(g2, cmap = 'gray')