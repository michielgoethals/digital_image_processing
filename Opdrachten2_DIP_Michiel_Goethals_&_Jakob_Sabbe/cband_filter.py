# dia 35 cband_filter
import numpy as np

def cband_filter(shape, D0, ftype='ideal', reject = True, W = 1, n=1):
    """
    Generate a circular band filter in the frequency domain.
    """
    r,c = shape
    D0 = np.atleast_1d(D0)
    K = D0.size
    W = np.ones((K,))*W
    n = np.ones((K,))*n
    R,C = np.ogrid[:r,:c]
    D = np.sqrt((R-int(r/2))**2 + (C -int(c/2))**2)
    H = np.ones(shape)
    
    for k in range(K):
        if ftype == 'ideal':
            H[(D0 - W/2 <= D) & (D <= D0 + W/2)] = 0.0
        elif ftype == 'gaussian':
            H = 1 - np.exp(-((D**2-D0**2)/(D*W))**2)
        elif (ftype == 'butterworth') | (ftype == 'btw'):
            H = 1 / (1 + ((D*W)/(D**2 - D0**2))**(2*n))
    
    if reject == False:
        H = np.abs(1-H)
    return H