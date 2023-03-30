import cv2
from cv2 import IMREAD_ANYCOLOR
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import SimilarityTransform, ProjectiveTransform, warp
from affineTransformations import get_matches, get_tf_model


def _get_stitch_images(im0,im1,tf_model=None,n_keypoints=500,min_samples=4,residual_threshold=2,**kwargs):
    if tf_model is None:
        tf_model = SimilarityTransform()
    elif tf_model == 'auto':
        tf_model = get_tf_model(im1,im0,xTransform=ProjectiveTransform,
                                n_keypoints=n_keypoints,min_samples=min_samples,
                                residual_threshold=residual_threshold, **kwargs)

    r,c = im0.shape[:2]
    corners0 = np.array([[0,0],[0,r],[c,0],[c,r]]) # hoeken van image 0
    r,c = im1.shape[:2]
    corners1 = np.array([[0,0],[0,r],[c,0],[c,r]]) # hoeken van image 1
    wcorners1= tf_model(corners1)                 # 
                  
    
    all_corners = np.vstack((corners0, corners1,wcorners1)) 
    print(all_corners)    
    min_corner = all_corners.min(axis=0)
    max_corner = all_corners.max(axis=0)
    print(min_corner) 
    print(max_corner) 
    new_shape = max_corner-min_corner
    print(new_shape)
    new_shape = np.ceil(new_shape[::-1]).astype(np.int)
    print(new_shape)
    shift = SimilarityTransform(translation=-min_corner)
    print(shift.params)
    print((tf_model+shift).inverse)
    im0_ = warp(im0,shift.inverse,output_shape=new_shape,cval=-1)
    im1_ = warp(im1,(tf_model+shift).inverse,output_shape=new_shape,cval=-1)
    return im0_,im1_

def _merge_stich_images(im0_,im1_,mask_index=None,cval=0):
    if mask_index is not None:
        if mask_index==0:
            im1_[im0_>-1] = 0
        elif mask_index ==1:
            im0_[im1_>-1] = 0
    else:
        alpha = 1.0*(im0_[:,:,0] !=-1)+1.0*(im1_[:,:,0]!=-1)
    bgmask = -1*((im0_==-1)&(im1_==-1))
    im0_[im0_==-1] = 0
    im1_[im1_==-1] = 0

    merged = im0_ + im1_
    if mask_index is None: merged /= np.maximum(alpha,1)[...,None]
    merged[bgmask==-1]=cval
    return merged

def _stitch(im0,im1, mask_idx=None,cval=0,show=True,tf_model=None,
            n_keypoints=500,min_samples=4,residual_threshold=2,**kwargs):
    im0_,im1_ = _get_stitch_images(im0,im1,tf_model=tf_model,
                                   n_keypoints=n_keypoints,min_samples=min_samples,residual_threshold=residual_threshold,**kwargs)
    merged = _merge_stich_images(im0_=im0_,im1_=im1_,mask_index=mask_idx,cval=cval)
    if (show):
        cv2.imshow('merged_image',merged)
    else:
        return merged


if __name__ == "__main__":
    
    src_im = cv2.imread('.\.\imgs\daenerys.jpg',IMREAD_ANYCOLOR) #0 means imread_grayscale
    dst_im = cv2.imread('.\.\imgs\\times-square.jpg',IMREAD_ANYCOLOR)
    
    r,c = src_im.shape[:2]
    src_corners = np.array([[0,0],[0,r],[c,0],[c,r]])
    
    plt.figure();plt.imshow(dst_im);plt.axis('off')     # plot src image
    dst_corners = np.array(plt.ginput(4,0))             # selecteer corners van dst image
    tf_model = ProjectiveTransform()                    # transfer_model is projectiveTransform
    tf_model.estimate(src_corners, dst_corners)         # doe transformatie van src_corners naar dst_corners
    src_tf_im = warp(src_im, tf_model.inverse)          # warp src_image via transformatie_model
    
    im0_, im1_ = _get_stitch_images(dst_im,src_im,tf_model=tf_model)
    merged = _merge_stich_images(im0_,im1_,mask_index=1)
    
    _stitch(dst_im,src_im,mask_idx=1,cval=0,show=True,tf_model=tf_model,
            n_keypoints=500,min_samples=4,residual_threshold=2)
    
    cv2.waitKey()


