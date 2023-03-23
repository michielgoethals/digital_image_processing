# dia 45: custom rectangular notch filter

import numpy as np
import matplotlib.pyplot as plt
import skimage
import imageio
from scipy import ndimage as ndi

def rnotch_filter(shape, D0=0, angle=0, ftype='ideal', reject=True, W=1, n=1):
    """
    Generate a rectangular notch filter in the frequency domain.
    shape: shape of the wilter,
    D0: start (from cneter) of the rectangular notch(es) (till image edge). (shape = (K,)).
    angle: angle (in degree) of notch with x-axis.
    ftype: type of the filter: 'ideal', 'gaussian' or 'butterworth'.
    reject: True of False (pass filter)
    W: notch width(s),
    n: order(s) of the butterworth filter notches
    """
    
    r,c = shape
    D0 = np.atleast_1d(D0)
    K = D0.size
    angle = np.ones((K,))*angle
    W = np.ones((K,))*W
    n = np.ones((K,))*n
    d = 2*np.ceil(np.sqrt((r/2)**2+(c/2)**2))+1
    R,C = [x-d//2 for x in np.ogrid[:d, :d]]
    
    D = np.abs(C) + np.ones((int(d),1))
    D2 = np.abs(R) + np.ones((int(d),1))
    
    H = []
    
    for k in range(K):
        if ftype == 'ideal':
            # Not working correctly
            Hk = np.where(((W/2) <= D2), 1,0)
            Hk2 = np.where((D0[k] <= D), 1,0)
            Hk = Hk | Hk2    
        elif ftype == 'gaussian':
            Hk = 1-np.exp(-0.5*((D**2)/(D0[k])**2))
            Hk2 = np.exp(-0.5*((D2**2)/(W)**2))
            Hk = Hk * Hk2
            Hk = np.abs(1-Hk)
        elif (ftype == 'butterworth') | (ftype == 'btw'):
            Hk = 1/( 1 + (D/D0[k])**(2*n))
            Hk2 = 1/( 1 + (D2/W)**(2*n))
            Hk = np.abs(1-Hk)  
            Hk = Hk * Hk2  
            Hk = np.abs(1-Hk)
            
        Hk = ndi.rotate(Hk,angle=angle[k],mode='reflect',reshape=False, order=1)
        
        Hk = Hk[int(d//2 - r//2):int(d//2 + r//2 + r%2), \
                int(d//2 - c//2):int(d//2 + r//2 + r%2)] 
        
        H.append(Hk)
    
    H = np.array(H).prod(axis=0)
    
    if reject == False:
        H = np.abs(1 - H)
    
    return H

if __name__ == "__main__":

    img = imageio.imread('./imgs/saturn_rings.tif') 
    img = skimage.util.img_as_float(img) 
    
    F = np.fft.fftshift(np.fft.fft2(img))
    
    H = rnotch_filter(img.shape, D0=10, ftype='btw', W=10, angle=90) 
    
    G = F*H
    g = np.real(np.fft.ifft2 (np.fft.fftshift(G))) 
    
    fig, (ax1, ax2,ax3,ax4) = plt.subplots (nrows=4, ncols=1, figsize=(3,12), sharex = True) 
    
    ax1.imshow(img, cmap ='gray') ;ax1.axis('off'); 
    ax1.set_title('Noisy image') 
    ax2.imshow(np.log(1+np.abs(F)), cmap ='gray') ;ax2.axis('off'); 
    ax3.set_title('Fourier spectrum') 
    ax3.imshow(np.log(1+np.abs(G)), cmap ='gray'); ax3.axis('off'); 
    ax3.set_title('Filtered Fourier spectrum') 
    ax4.imshow(g, cmap ='gray') ;ax4.axis ('off'); 
    ax4.set_title('Filtered image')
