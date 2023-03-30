import cv2
import numpy as np
from projectiveTransformations import _stitch
from skimage.transform import rescale

import matplotlib.pyplot as plt

def stitch(ims, order=[1,0,2], mask_idx=None, 
           tf_model ='auto', n_keypoints=1000,
           min_samples=4, residual_threshold=2, **kwargs):
    
    # sort images according to order
    
    ims_sorted = np.zeros_like(ims)
    
    for i, idx in enumerate(order):
        print(i)
        print(idx)
        ims_sorted[idx] = ims[i]
    
    # apply _stitch
    
    
    
    # repeat _stitch with merged and next im until cval =0
    
    return ims_sorted


if __name__ == "__main__":
    files=['DFM_4209.jpg','DFM_4210.jpg','DFM_4211.jpg']
    ims = []
    for i,file in enumerate(files):
        im = cv2.imread('../../imgs/'+file ,cv2.IMREAD_ANYCOLOR)                 # inlezen van alle foto's
        im = im[:,500:500+1987,:]                                           # foto bijsnijden voor beter stiching
        ims.append(rescale(im,0.25,anti_aliasing=True,multichannel=True))   # elke foto toegevoegd aan ims array

    merged = stitch(ims)
    
    
    plt.figure();plt.imshow(merged[2]);plt.axis('off') 