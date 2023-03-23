from numpy.fft import fft2, ifft2, fftshift
import cv2
import numpy as np
import skimage
import matplotlib.pyplot as plt


def linearmotionblurfilter(shape, length=1, angle=0, domain='spatial'):
    r,c = shape
    d = int(np.sqrt(r**2 + c**2))
    
    psf = np.zeros((d,d))
    psf[d//2,d//2:d//2+length] = 1
    
    psf = skimage.transform.rotate(psf, angle)
    
    psf = crop2center(psf,shape)
    psf = np.fft.fftshift(psf)
    if (domain == 'freq') or (domain == 'frequency'):
        psf = np.fft.fftshift(np.fft.fft2(psf))
        
    return psf


#eigen
'''  
def crop2center(image, shape):
    a,b = image.shape
    r,c = shape
    x = (a-r)//2
    y = (b-c)//2
    return image[x:-x,y:-y]
'''

#roeland
def crop2center(psf,shape):
    width, height = psf.shape
    new_height, new_width = shape

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))


    center_cropped_img = psf[top:bottom, left:right, ...]

    return center_cropped_img


if __name__ == "__main__":
    
    img = cv2.imread('.\imgs\daenerys.jpg',cv2.IMREAD_GRAYSCALE)/255
    img_f = fftshift(fft2(img))
    psf = linearmotionblurfilter(img.shape,length=100,angle=-30)
    blurred = np.real(ifft2(fftshift(img_f*psf)))
    
    fig, axes = plt.subplots(ncols=3, figsize=(19, 4))
    ax = axes.ravel()
    [axi.set_axis_off() for axi in ax.ravel()]
    
    
    
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(blurred, cmap='gray')
    ax[1].set_title('motion blur')
    ax[2].imshow(psf, cmap='gray')
    ax[2].set_title('motion blur')
    


