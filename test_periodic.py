# Test of periodic_noise function
from skimage.morphology import square, dilation

img = cv2.imread('./imgs/Apollo17boulder.tif', cv2.IMREAD_GRAYSCALE)/255

fig, axs = plt.subplots(ncols=3, figsize=(12,6))
ax = axs.ravel()
[axi.set_axis_off() for axi in ax.ravel()]

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')

C = np.array([[0, 44],[44,0],[44,44],[44,-44]]) 

r, R = periodic_noise(img.shape, C, [1,1,1,1])
g = img + r/3
G = np.abs(np.fft.fftshift(np.fft.fft2(g)))
Gd =dilation(np.log(1+G),square(3))

ax[1].imshow(Gd, cmap='gray')
ax[1].set_title('Frequency noise R')

ax[2].imshow(g, cmap='gray')
ax[2].set_title('spatial noise r')
