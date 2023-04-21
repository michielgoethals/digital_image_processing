import imageio
import cv2
from piecewiseWarping import warp_image
import numpy as np
from scipy.spatial import Delaunay
from featureDetectionDlib import feature_detection_dlib
import matplotlib.pyplot as plt
from ConvexHull import convexHull, create_mask, pyramid_blend
from pyramidBlending import get_laplacian_pyramid, get_gaussian_pyramid, pyramid_blend, plot_pyramid
import os

def show_swap_faces(img1,img2=None,blendmode='pyramid',faceorder=(0,1),flip_faces=(True,True),detail='convexhull',plot_Delaunay_Keypoints='True', Title = 'FaceSwap'):
    img = swap_faces(img1,img2,blendmode,faceorder,flip_faces,detail,plot_Delaunay_Keypoints)
    plt.imshow(img)
    plt.title(Title)
    plt.show()


def swap_faces(img1,img2=None,blendmode='pyramid',faceorder=(0,1),flip_faces=(True,True),detail='convexhull',plot_Delaunay_Keypoints='True'):
    """_summary_
    Swap face in image 1 with face in image 2 using the specified blendmode. If image 2 = None then image 1 should contain two faces and the faceorder arguments specifies the swapping order.

    Args:
        img1 (_type_): Image 1
        img2 (_type_, optional): Image 2. Defaults to None.
        blendmode (str, optional): Mode of blending of images can be pyramid, cv and alpha. Defaults to 'pyramid'.
        faceorder (tuple, optional): Order of faces swaped when image 1 has 2 faces. Defaults to (0,1).
        flip_faces (tuple, optional): Flips the orientation of the faces. Defaults to (True,True). Only works for two separate images.
        detail (str, optional): Warp with convex hull or full feature warp. Defaults to 'convexhull'. Can also be full feature warp.
        plot_Delaunay_Keypoints (str, optional): If wanted the keypoints and Delauney triangles are plotted for the images. Defaults to 'True'.

    Returns:
        Faceswapped image: Return the image of the face of image 2 placed on the body of image 1 with the required blending technique in place
    """
    r, c = img1.shape[:2]

    if img2 is None:
        if faceorder[0] == 0:       # split image 1 in two halves and hope the faces are in a different half
            img2 = img1[:,c//2:c]   
            img1 = img1[:,0:c//2]
        else:
            img2 = img1[:,0:c//2]
            img1 = img1[:,c//2:c]
        img1s = swap_faces(img1,img2,blendmode=blendmode,plot_Delaunay_Keypoints=plot_Delaunay_Keypoints,detail=detail,flip_faces=(False,flip_faces[0]))
        img2s = swap_faces(img2,img1,blendmode=blendmode,plot_Delaunay_Keypoints=plot_Delaunay_Keypoints,detail=detail,flip_faces=(False,flip_faces[1]))
        swapped = np.hstack([img1s,img2s])
        return swapped
    
    else:
        pwd = os.path.dirname(__file__)
        model = pwd + "/data/facelandmarks/shape_predictor_68_face_landmarks.dat"

        # flip the faces if needed
        if flip_faces[0]:
            img1 = np.fliplr(img1)
        if flip_faces[1]:
            img2 = np.fliplr(img2)
        
        if detail == 'convexhull':
            con1,tri1,conf1,vert1 = convexHull(img1,model=model,DEBUG=plot_Delaunay_Keypoints)
            con2,tri2,conf2,vert2 = convexHull(img2,model=model,DEBUG=plot_Delaunay_Keypoints)
            div = len(con1)-len(con2)
            print("div: "+str(div))
            if div > 0:
                con1,tri1,conf1,vert1 = convexHull(img1,model=model,limited=vert2,DEBUG=plot_Delaunay_Keypoints)
            elif div < 0:
                con2,tri2,conf2,vert2 = convexHull(img2,model=model,limited=vert1,DEBUG=plot_Delaunay_Keypoints)
            warped = warp_image(img2, conf2, tri2,conf1, img1.shape)

            if plot_Delaunay_Keypoints == True:
                plt.figure()
                plt.title("warped")
                plt.imshow(warped)
        else:
            pts1 = feature_detection_dlib(img1,model, True) 
            pts2 = feature_detection_dlib(img2,model, True)
            # print(len(pts1))
            triangles2 = Delaunay(pts2)
            warped = warp_image(img2, pts2, triangles2,pts1, img1.shape)


        con1,tri1,conf,vert = convexHull(warped,model=model)            # bereken de convexhull van de featurepoints van img1
        mask = create_mask(warped.shape, con1)[:r,:c]                   # maak een masker van de convexhull van img1
        warped2 = warped[:r,:c]                                         # herschaal het masker naar de grootte van img1
                                                

        if blendmode == 'pyramid':
            gmask = get_gaussian_pyramid(mask*255)                      # maak een gaussian pyramid van het masker
            Limage1 = get_laplacian_pyramid(img1)                       # maak een laplacian pyramid van img1
            Limage2 = get_laplacian_pyramid(warped2*255)                # maak een laplacian pyramid van img2
            blended = pyramid_blend(gmask,Limage1,Limage2)              # blend de laplacian pyramid met de gaussian pyramid en recontrueer de image
            return blended
        
        if blendmode == 'cv':
            img1c = np.uint8(img1)
            img2c = np.uint8(warped2*255)
            mask3 = np.uint8(mask*255)
            center = (c//2,r//2)
            swap_cv2 = cv2.seamlessClone(img2c,img1c,mask3,center, cv2.NORMAL_CLONE)
            return swap_cv2
        
        if blendmode == 'alpha':
            blended = img1/255*(1-mask) + warped2*mask
            return blended




if __name__ == "__main__":
    image_folder = "././imgs/faces/"

    img_name = "gal_gadot.jpg" 
    galgadot = imageio.imread(image_folder+img_name)            
    img_name = "nicolas_cage.jpg" 
    nickcage = imageio.imread(image_folder+img_name)
    img_name = "brangelina.jpg" 
    brangelina = imageio.imread(image_folder+img_name)
    img_name = "hillary_clinton.jpg"
    hillary = imageio.imread(image_folder+img_name)
    img_name = "donald_trump.jpg"
    trump = imageio.imread(image_folder+img_name)
    img_name = "daenerys.jpg"
    daenerys = imageio.imread(image_folder+img_name)



    show_swap_faces(galgadot,nickcage,blendmode='pyramid',plot_Delaunay_Keypoints=True,detail='feature',flip_faces=(False,True), Title="pyramid blend")
    show_swap_faces(galgadot,nickcage,blendmode='cv',plot_Delaunay_Keypoints=True,detail='feature',flip_faces=(False,True), Title="cv2 blend")
    show_swap_faces(galgadot,nickcage,blendmode='alpha',plot_Delaunay_Keypoints=True,detail='feature',flip_faces=(False,True), Title="alpha blend")
    show_swap_faces(brangelina,blendmode='pyramid',plot_Delaunay_Keypoints=True,detail='Full feature', Title="One picture")
    show_swap_faces(hillary,trump,blendmode='cv',plot_Delaunay_Keypoints=False,detail='Full feature')
    show_swap_faces(hillary,trump,blendmode='pyramid',plot_Delaunay_Keypoints=False,detail='Full feature')
    show_swap_faces(daenerys,galgadot,blendmode='pyramid',plot_Delaunay_Keypoints=False, detail='Full feature', Title="full feature")
    show_swap_faces(daenerys,galgadot,blendmode='pyramid',plot_Delaunay_Keypoints=False, detail='convexhull', Title="convexhull")



