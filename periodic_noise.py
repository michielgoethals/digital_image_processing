# dia 29 Custom periodic_noise function

import numpy as np
from scipy import signal

def periodic_noise(shape, C, A = None, B = None):
    """
    Generate periodic noise arrays (r:spatial domain, R:frequency domain).
    shape:shape of array
    C: (K,2)array of energy burst frequencies
    A: (K,)vector of burst magnitudes
    B: (K,2)array of phase shifts (Bx,By)
    """
    
    M, N = shape
    
    # K = pairs of freq domain coordinates (u,v)
    K = C.shape[0]
    
    if A is None:
        A = np.ones((K,))
        B = np.zeros((K, 2))
        B[1:K, 0:2] = 0
    if B is None:
        B = np.zeros((K, 2))
        B[1:K, 0:2] = 0
    
    u, v = np.mgrid[:M,:N].astype('float32')
    
    R = np.zeros((M,N), dtype = complex)
    
    for i in range(K):
        u0, v0 = C[i]
        Bx, By = B[i]
        R = 1j * A[i]/2 * M * N * (
            np.exp(-1j * 2 * np.pi * (u0 * Bx / M + v0 * By / N) * signal.unit_impulse((u+u0, v+v0))) -
            np.exp(1j * 2 * np.pi * (u0 * Bx / M + v0 * By / N) * signal.unit_impulse((u-u0, v-v0)))
         )
        
    r = np.real(np.fft.ifft2(np.fft.ifftshift(R)))
    
    return r, R

