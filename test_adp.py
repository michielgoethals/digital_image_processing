import cv2
from skimage.util.noise import random_noise
from adpfilter import adpmedian
from matplotlib import pyplot as plt

img = cv2.imread('./imgs/ckt-board-saltpep.tif', cv2.IMREAD_GRAYSCALE)

fig, axs = plt.subplots(ncols=2, figsize=(12,6))
ax = axs.ravel()
[axi.set_axis_off() for axi in ax.ravel()]

ax[0].imshow(img, cmap='gray', vmin = 1.0, vmax = 255)
ax[0].set_title('Noisy Image')

filtered = adpmedian(img, 9)

ax[1].imshow(filtered, cmap='gray', vmin = 1.0, vmax = 255)
ax[1].set_title('Filtered with function')