# dia 20 Adaptive median spatial filtering
import numpy as np

def adpmedian(g, Smax):
    """
    Perform adaptive median filtering on image g (maximum neighbourghood size = Smax).
    """
    y, x = g.shape
    f = np.zeros((y, x), dtype=np.uint8)
    
    for i in range(y):
        for j in range(x):
            Sxy = 3
            while Sxy <= Smax:
                window = g[max(0, i-Sxy//2):min(y, i+Sxy//2+1),
                              max(0, j-Sxy//2):min(x, j+Sxy//2+1)]
                Zmed = np.median(window)
                Zmax = np.max(window)
                Zmin = np.min(window)
                Zxy = g[i, j]
                
                A1 = Zmed - Zmin
                A2 = Zmed - Zmax
                
                if A1 > 0 and A2 < 0:
                    B1 = Zxy - Zmin
                    B2 = Zxy - Zmax
                    if B1 > 0 and B2 < 0:
                        # output Zxy
                        f[i, j] = Zxy
                    else:
                        f[i, j] = Zmed
                    break
                else:
                    # increase window size
                    Sxy += 1
            else:
                f[i, j] = Zmed
    return f

