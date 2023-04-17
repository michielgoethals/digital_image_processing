import imageio
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from scipy.spatial import Delaunay
from skimage.transform import warp, AffineTransform
from skimage.draw import polygon
from featureDetectionDlib import feature_detection_dlib


def get_bounding_box(pts):
    # X coordinate is point[0], Y coordinate is point[1]
    top_left_x = int(min(point[0] for point in pts))
    top_left_y = int(min(point[1] for point in pts))
    
    width = int(max(point[0] for point in pts)) - top_left_x
    height = int(max(point[1] for point in pts)) - top_left_y
    
    # Return int array with (x,y) coordinates of top left corner, width and height
    return [top_left_x, top_left_y, width, height]

def warp_triangle(img, bbox, tf_M, output_shape):
    
    x0, y0, width, height = bbox 
    
    xmax = x0 + width
    ymax = y0 + height
    
    # Select all pixels from the bounding box 
    sub_img = img[y0:ymax, x0:xmax]
    
    transformed = warp(sub_img, tf_M, output_shape=output_shape, mode='reflect')    
    #mode parameter -> {constant, edge, symmetric, reflect, wrap},
    return transformed
    
def get_triangle_mask(triangle_mask, bbox, output_shape):
    mask = np.zeros(output_shape, dtype=np.uint8)
    r,c = polygon(triangle_mask[:, 0]-(bbox[0] + 1), triangle_mask[:, 1] - (bbox[1] + 1))
    mask[c,r] = 1
    mask = mask.reshape(*mask.shape, 1)
    return mask
    
def warp_image(img, points, triangles, points_m, imageShape):

    warped = np.zeros((max(img.shape[0], imageShape[0]), max(img.shape[1], imageShape[1]),3))
    
    for tri in triangles.simplices:
        t1 = np.array([points[tri[0]], points[tri[1]], points[tri[2]]],dtype=int)
        tm = np.array([points_m[tri[0]], points_m[tri[1]], points_m[tri[2]]], dtype=int)

        
        bb1 = get_bounding_box(t1)
        bbm = get_bounding_box(tm)
    

        M = AffineTransform()
        M.estimate(t1-bb1[:2],tm-bbm[:2])        
        
        if not np.linalg.det(M.params): 
            continue
        else: 
            M = np.linalg.inv(M.params)
        
        x0, y0, width, height = bbm 
    
        xmax = x0 + width
        ymax = y0 + height
        
        output_shape = warped[y0:ymax, x0:xmax].shape[:2]
        
        wt1 = warp_triangle(img, bb1, M, output_shape)
        
        mask = get_triangle_mask(tm, bbm, output_shape)

        warped[y0:ymax, x0:xmax] = warped[y0:ymax, x0:xmax]*(1-mask)+ mask*wt1
        
    return warped

if __name__ == "__main__":
    
    im1 = "../../imgs/faces/daenerys.jpg"   # Path to first image
    im2 = "../../imgs/faces/gal_gadot.jpg"       # Path to second image
    
    img1 = imageio.imread(im1)                  # Load the first image
    img1 = skimage.util.img_as_float(img1)      # Convert image data type

    img2 = imageio.imread(im2)                  # load the second image
    img2 = skimage.util.img_as_float(img2)      # Convert image data type
    
    # Plot the images
    plt.subplot(121);plt.title('Image 1',fontsize=20);plt.axis('off');plt.imshow(img1)
    plt.subplot(122);plt.title('Image 2',fontsize=20);plt.axis('off');plt.imshow(img2)
    
    # Path to landmark
    model68 = "./data/facelandmarks/shape_predictor_68_face_landmarks.dat" 
    
    pts1 = feature_detection_dlib(img1, model68, corners=True)
    pts2 = feature_detection_dlib(img2, model68, corners=True)
    
    triangles1 = Delaunay(pts1)      # Get list of triangles (image1) 
    triangles2 = Delaunay(pts2)      # Get list of triangles (image2)

    # Input alpha value
    alpha = 0.5
    
    # Calculate intermediate points                                 
    ptsm = (1-alpha)*pts1 + alpha*pts2                                         
    
    # Warped images
    warped1 = warp_image(img1, pts1, triangles1, ptsm, img2.shape)  # Warp image1                                
    warped2 = warp_image(img2, pts2, triangles2, ptsm, img1.shape)  # Warp image2
        
    # Morphed image
    morphed = img_as_ubyte((1-alpha)* warped1 + alpha * warped2)
    
    # Calculate new width and height of the warped images
    newWidth = int((1-alpha)*img1.shape[0] + alpha*img2.shape[0])       
    newHeight = int((1-alpha)*img1.shape[1] + alpha*img2.shape[1])
    
    plt.subplot(131);plt.title('Warped 1',fontsize=30);plt.imshow(warped1[0:newWidth, 0:newHeight]);plt.axis('off')
    plt.subplot(132);plt.title('Warped 2',fontsize=30);plt.imshow(warped2[0:newWidth, 0:newHeight]);plt.axis('off')   
    plt.subplot(133);plt.title('Morphed images',fontsize=30);plt.imshow(morphed[0:newWidth, 0:newHeight]);plt.axis('off')      
                  

