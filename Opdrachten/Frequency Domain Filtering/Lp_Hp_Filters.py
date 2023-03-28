#dia 46: custom LPC and HPC filter functions
import numpy as np

def lpcfilter(shape, ftype='ideal', D0 = 0, n=1, center=(0,0)):
    """
    Generate low pass circular filter H in the frequency domain
    shape: shape of the filter, 
    ftype: type of the filter: 'ideal', 'gaussian' or 'butterworth', 
    DO: range of the filter frame
    n: order of the butterworth filter, 
    center: shift of the center point of the filter relative to the center of the frequency rectangle 
    """
    r,c = shape
    R,C = np.ogrid[:r,:c]
    H = np.zeros((r,c))
    D = np.sqrt((R -  r//2 - center[0])**2 + (C -  c//2 - center[1])**2)
    
    if ftype == 'ideal':
        #all pixels within radius set to 1.0
        H[D < D0] = 1.0
    elif ftype == 'gaussian':
        H = np.exp(-(D)**2/(2*(D0)**2))
    elif (ftype == 'butterworth') | (ftype == 'btw'):
        H = 1/(1 + (D/D0)**(2*n))
    
    return H
    
def hpcfilter(shape, ftype='ideal', D0 = 0, n=1, center=(0,0)):
    """
    Generate low pass circular filter H in the frequency domain
    """
    H = 1 - lpcfilter(shape, ftype = ftype, D0 = D0, n = n, center = center) 
    
    return H
