# dia 45: custom rectangular notch filter

import numpy as np

def rnotch_filter(shape, D0=0, angle=0, ftype='ideal', reject=True, W=1, n=1):
    r,c = shape
    D0 = np.atleast_1d(D0)
    K = D0.size
    
    angle = np.ones((K,))*angle
    W = np.ones((K,))*W
    n = np.ones((K,))*n
    
    d = 2*np.ceil(np.sqrt((r/2)**2+(c/2)**2))+1
    R,C = [x-d//2 for x in np.ogrid[:d, :d]]
    
    H = []
    
    for k in range(K):
        if ftype == 'ideal':
            Hk = np.ones((d,d))
            center = (d-1)//2
            for i in range(d):
                for j in range(d):
                    if ((i-center)**2 + (j-center)**2)**0.5 <= D0[k]:
                        Hk[i,j] = 0
        elif ftype == 'gaussian':
            Hk = np.exp(-((R**2+C**2)/(2*W[k]**2)))*np.ones((d,d))
            center = (d-1)//2
            for i in range(d):
                for j in range(d):
                    if ((i-center)**2 + (j-center)**2)**0.5 <= D0[k]:
                        Hk[i,j] = 1-Hk[i,j]

        elif (ftype == 'butterworth') | (ftype == 'btw'):
            Hk = np.ones((d,d))
            center = (d-1)//2
            for i in range(d):
                for j in range(d):
                    if ((i-center)**2 + (j-center)**2)**0.5 != 0:
                        Hk[i,j] = 1/(1+((D0[k]*W[k])/(((i-center)**2 + (j-center)**2)**0.5)**(2*n[k])))

            
        Hk = ndi.rotate(Hk,angle=angle[k],mode='reflect',reshape=False, order=1)
        
        Hk = Hk[int(d//2 - r//2):int(d//2 + r//2 + r%2), \
                int(d//2 - c//2):int(d//2 + r//2 + r%2)] 
        
        H.append(Hk)
    
    H = np.array(H).prod(axis=0)
    
    if reject == False:
        H = np.abs(1 - H)
    
    return H
