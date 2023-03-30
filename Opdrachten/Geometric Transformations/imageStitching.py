import cv2
import numpy as np
from projectiveTransformations import _stitch
import skimage
from skimage.transform import rescale


import matplotlib.pyplot as plt

def stitch(ims, order=[1,0,2], mask_idx=None, 
           tf_model ='auto', n_keypoints=1000,
           min_samples=4, residual_threshold=2, **kwargs):
    
    # sort images according to order
    
    ims_sorted = np.zeros_like(ims)
    
    for i, idx in enumerate(order):
        ims_sorted[idx] = ims[i]
    
    # apply _stitch
    
    new_img = _stitch(ims_sorted[0],ims_sorted[1],mask_idx=1,cval=-1,show=False,tf_model=tf_model,
        n_keypoints=500,min_samples=4,residual_threshold=2)
    
    ims_sorted = _stitch(new_img,ims_sorted[2],mask_idx=1,cval=0,show=False,tf_model=tf_model,
        n_keypoints=500,min_samples=4,residual_threshold=2)
    
    
    # repeat _stitch with merged and next im until cval =0
    
    return ims_sorted


if __name__ == "__main__":
    files=['DFM_4209.jpg','DFM_4210.jpg','DFM_4211.jpg']
    ims = []
    for i,file in enumerate(files):
        im = cv2.imread('././imgs/'+file ,cv2.IMREAD_ANYCOLOR)                 # inlezen van alle foto's
        im = im[:,500:500+1987,:]                                              # foto bijsnijden voor beter stiching
        #Rescale -> vanaf versie skimage v0.19 is multichannel veranderd naar channel_axis                                
        ims.append(rescale(im,0.25,anti_aliasing=True,channel_axis=2))   # elke foto toegevoegd aan ims array

    merged = stitch(ims)
    
    
    #plt.figure()
    #plt.imshow(merged);plt.axis('off')
    #plt.show()
    cv2.imshow("tse", merged)
    input("Press Enter to continue...")
