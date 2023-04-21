import imageio
import cv2
from piecewiseWarping import warp_image
import numpy as np
from scipy.spatial import Delaunay
from featureDetectionDlib import feature_detection_dlib
import matplotlib.pyplot as plt
from ConvexHull import convexHull, create_mask, pyramid_blend
from pyramidBlending import get_laplacian_pyramid, get_gaussian_pyramid, pyramid_blend
import os


def swap_faces(img1,img2=None,blendmode='pyramid',faceorder=(0,1),flip_faces=(True,True),detail='convexhull',plot_Delaunay_Keypoints='True'):
    """_summary_
    Swap face in image 1 with face in image 2 using the specified blendmode. If image 2 = None then image 1 should contain two faces and the faceorder arguments specifies the swapping order.

    Args:
        img1 (_type_): Image 1
        img2 (_type_, optional): Image 2. Defaults to None.
        blendmode (str, optional): Mode of blending of images can be alpha,pyramid or cv. Defaults to 'pyramid'.
        faceorder (tuple, optional): Order of faces swaped when image 1 has 2 faces. Defaults to (0,1).
        flip_faces (tuple, optional): Flips the orientation of the faces. Defaults to (True,True).
        detail (str, optional): Warp with convex hull or full feature warp. Defaults to 'convexhull'. Can also be full feature warp.
        plot_Delaunay_Keypoints (str, optional): If wanted the keypoints and Delauney triangles are plotted for the images. Defaults to 'True'.

    Returns:
        Faceswapped image: Return the image of the face of image 2 placed on the body of image 1 with the required blending technique in place
    """

    if img2 is None:
        r,c,n = img1.shape
        if faceorder[0] == 0:
            img2 = img1[:,c//2:c]   
            img1b = img1[:,0:c//2]
        else:
            img2 = img1[:,0:c//2]
            img1b = img1[:,c//2:c]

        img1s = swap_faces(img1b,img2,blendmode=blendmode,plot_Delaunay_Keypoints=plot_Delaunay_Keypoints,detail=detail,flip_faces=(False,flip_faces[0]))
        plt.figure()
        plt.imshow(img1s)
        img2s = swap_faces(img2,img1b,blendmode=blendmode,plot_Delaunay_Keypoints=plot_Delaunay_Keypoints,detail=detail,flip_faces=(False,flip_faces[1]))
        plt.figure()
        plt.imshow(img2s)
        swapped = np.hstack([img1s,img2s])
        return swapped
    
    else:
        if isinstance(img1[2][0][0], np.uint8):
            img1 = img1/255
        if isinstance(img2[2][0][0], np.uint8):
            img2 = img2/255

        if flip_faces[0] == True:
            img1 = np.fliplr(img1)
        if flip_faces[1] == True:
            img2 = np.fliplr(img2)

        pwd = os.path.dirname(__file__)
        model = pwd + "/data/facelandmarks/shape_predictor_68_face_landmarks.dat"
        
        if detail == 'convexhull':
            con1,tri1,conf1,vert1 = convexHull(img1,model=model,DEBUG=plot_Delaunay_Keypoints)
            con2,tri2,conf2,vert2 = convexHull(img2,model=model,DEBUG=plot_Delaunay_Keypoints)
            # print(len(con1))
            # print(len(con2))
            div = len(con1)-len(con2)
            print("div: "+str(div))
            if div > 0:
                con1,tri1,conf1,vert1 = convexHull(img1,model=model,DEBUG=plot_Delaunay_Keypoints)
            elif div < 0:
                con2,tri2,conf2,vert2 = convexHull(img2,model=model,DEBUG=plot_Delaunay_Keypoints)
            # print(len(tri1.simplices))
            # print(len(tri2.simplices))
            try:
                warped2 = warp_image(img2, conf2, tri2,conf1, img1.shape)
            except:
                print('Image does not have enough recognisable faces')
                return 
            if plot_Delaunay_Keypoints == True:
                plt.figure()
                plt.title("warped2")
                plt.imshow(warped2)
        else:
            pts1 = feature_detection_dlib(img1,model, True) 
            pts2 = feature_detection_dlib(img2,model, True)
            # print(len(pts1))
            triangles2 = Delaunay(pts2)
            # print(len(triangles2.simplices))
            try:
                warped2 = warp_image(img2, pts2, triangles2,pts1, img1.shape)
            except:
                print('Image does not have enough recognisable faces')
                return 
        con1,tri1,conf1,vert = convexHull(warped2,model=model,DEBUG=plot_Delaunay_Keypoints)
            
        mask = create_mask(warped2.shape,con1)
        r,c,n = img1.shape
        mask2 = mask[:r,:c]         
        warp = warped2[:r,:c]
        if blendmode == 'pyramid':
            gmask = get_gaussian_pyramid(mask2)
            Limg1 = get_laplacian_pyramid(img1)
            Lwarp = get_laplacian_pyramid(warp)
            swap_pyramid = pyramid_blend(gmask,Limg1,Lwarp)
            return swap_pyramid
        elif blendmode == 'alpha':
            face = warp * mask2
            body = img1 * (1-mask2)
            swap_alpha = body+face
            return swap_alpha 
        elif blendmode == 'cv':
            # print(type(img1[0][0][0]))
            img1c = np.uint8(img1*255)
            img2c = np.uint8(warp*255)
            mask3 = np.uint8(mask2*255)
            center = (c//2,r//2)
            swap_cv2 = cv2.seamlessClone(img2c,img1c,mask3,center, cv2.NORMAL_CLONE)
            return swap_cv2
        else:
            print('error wrong blendmode try: pyramid,alpha or cv')

if __name__ == "__main__":
    image_folder = "././imgs/faces/"
    img_name = "gal_gadot.jpg" 
    galgadot = imageio.imread(image_folder+img_name)            

    plt.imshow(galgadot)    
    plt.show()
    img_name = "nicolas_cage.jpg" 
    nickcage = imageio.imread(image_folder+img_name)       

    plt.imshow(nickcage)
    plt.show()

    swapped = swap_faces(galgadot,nickcage,blendmode='pyramid',plot_Delaunay_Keypoints=True,detail='convexhull',flip_faces=(False,True))
    plt.figure()
    plt.imshow(swapped)







