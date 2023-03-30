import numpy as np
import cv2
from cv2 import IMREAD_ANYCOLOR, KeyPoint, imshow
from skimage.transform import warp, AffineTransform
from skimage.transform import AffineTransform
from skimage.feature import (match_descriptors, ORB, plot_matches)
from skimage.measure import ransac
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

def get_matches(im_or, im_tf, n_keypoints=500,ax = None, title ='Original vs transformed'):
    descriptor_extractor = ORB(n_keypoints=n_keypoints)

    #zoekt naar keypoints en discroptions in orignal image
    descriptor_extractor.detect_and_extract(rgb2gray(im_or))
    keyPoint_or = descriptor_extractor.keypoints
    descriptors_or = descriptor_extractor.descriptors

    #zoekt naar keypoints en discroptions in TF image
    descriptor_extractor.detect_and_extract(rgb2gray(im_tf))
    keypoints_tf = descriptor_extractor.keypoints
    descriptors_tf = descriptor_extractor.descriptors

    #Zoekt matches tussen de twee images
    matches = match_descriptors(descriptors_or,descriptors_tf,cross_check=True)

    if ax is not None:
        plot_matches(ax,im_or,im_tf,keyPoint_or,keypoints_tf, matches)
        ax.axis('off')
        ax.set_title(title)
    return matches, keyPoint_or, keypoints_tf

# maken van een transfermodel via ransac functie.
def get_tf_model(src,dst, xTransform=AffineTransform, n_keypoints=500,min_samples=4,residual_threshold=2,**kwargs):
    matches,kp_src, kp_dst = get_matches(src,dst,n_keypoints=n_keypoints)
    src = kp_src[matches[:,0]][:,::-1]
    dst = kp_dst[matches[:,1]][:,::-1]
    tf_model, _ =ransac((src,dst),xTransform,min_samples=min_samples,residual_threshold=residual_threshold, **kwargs)
    return tf_model


if __name__ == "__main__":
    

    im = cv2.imread('..\..\imgs\yoda.jpg', IMREAD_ANYCOLOR) 
    cv2.imshow('image',im)
    c = np.array(im.shape[:2])//2
    
    T = np.diag([1,1,1])
    T[:2,-1] = -c[::-1]
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    Ti = np.diag([1,1,1])
    Ti[:2,-1] = c[::-1]
    A = np.dot(Ti,np.dot(R,T))
    
    imA= warp(im,np.linalg.inv(A),order=3)
    cv2.imshow('image',imA)
    
    fig1, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(20,10))
    matches, kp_or, kp_tf = get_matches(im,imA,n_keypoints=200, ax=ax1)
    
    tf_model = get_tf_model(im,imA,xTransform=AffineTransform,n_keypoints=200,min_samples=4,residual_threshold=2)
    print(tf_model.params)
    plt.show()
    