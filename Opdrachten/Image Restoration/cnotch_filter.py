# dia 41: cnotch_filter

import numpy as np

from Lp_Hp_Filters import hpcfilter, lpcfilter

def cnotch_filter(shape, centers, ftype='ideal', reject=True, D0=0, n=1):
    """
    Generate a circular notch filter in the frequency domain.
    shape:shape of the filter,
    centers: notch frequency coordinates(shape=(K,2))
    ftype: type of the filter: 'ideal' , 'gaussian', or 'butterworth' 
    reject: True or false (pass filter),
    D0: notch size(s),
    n: order(s) of the butterworth filter notches
    """
    
    K = centers.shape[0]
    H = np.ones(shape)
    
    for k in range(K):
        Hk = hpcfilter(shape, ftype=ftype, D0=D0, n=n, center=centers[k])
        H_k = hpcfilter(shape,ftype=ftype, D0=D0, n=n, center=-centers[k])
        H *= (Hk*H_k)         
    
    if reject == False:
        H = np.abs(1-H)
    return H