import dlib
import numpy as np
from imutils import face_utils
import imageio
import skimage
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.color import rgb2gray
import cv2
import os


def add_corners(pts, img):
    height,width,depth = img.shape
    
    TL = [0,0]          # Top left corner
    TR = [0,height]     # Top right corner
    BL = [width,0]      # Bottom left corner
    BR = [width,height] # Bottom right corner  

    TM = [int(width/2),0]       # Top midpoint
    BM = [int(width/2),height]  # Bottom midpoint
    LM = [0,int(height/2)]      # Left midpoint
    RM = [width,int(height/2)]  # Right midpoint 
    
    cornersAndMidpoints = np.array([TL,TR,BL,BR, LM, RM, TM, BM])  # Place all above points in array

    pts = np.vstack((pts, cornersAndMidpoints))
    
    return pts

  
def feature_detection_dlib(img, model: str, corners=True):
    p = model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    
    gray = img_as_ubyte(rgb2gray(img))
    rects = detector(gray, upsample_num_times = 0)
    
    pts = []

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        pts.append(shape)
        
    pts = pts[0] # This creates a numpy array
    
    if corners:
        pts = add_corners(pts, img)
    
    
    return pts
    

if __name__ == "__main__":
    
    path_to_img = "././imgs/faces/nicolas_cage.jpg"
    

    image = imageio.imread(path_to_img)    # load the image
    image = skimage.util.img_as_float(image)                        # Convert image data type
    
    # path to landmarks
    # path to landmarks
    pwd = os.path.dirname(__file__)
    model68 = pwd + "/data/facelandmarks/shape_predictor_68_face_landmarks.dat"  
    model5 =  pwd + "/data/facelandmarks/shape_predictor_5_face_landmarks.dat"
    
    #predict points
    pts68 = feature_detection_dlib(image, model68, corners=True)
    pts5 = feature_detection_dlib(image, model5, corners=True)
    
    radius = 4          # Radius of circle points
    color = (255, 0, 0) # Red color of points on image
    thickness = -1      # Color the inside of circles

    pts68 = pts68.reshape((-1, 2)) # Reshape because a np.int32 is not iterable
    for (x, y) in pts68:
        cv2.circle(image, (x, y), radius, color, thickness)     # Draw a circle on all points
    
    image5 = imageio.imread(path_to_img)        # load the image again
    image5 = skimage.util.img_as_float(image5)  # Convert image data type 
    
    pts5 = pts5.reshape((-1, 2))
    for (x, y) in pts5:
          cv2.circle(image5, (x, y), radius, color, thickness)
          

    fig, axes = plt.subplots(ncols=2, figsize=(20, 5))
    ax = axes.ravel()
    [axi.set_axis_off() for axi in ax.ravel()]
    ax[0].imshow(image, cmap='gray', vmin = 1.0, vmax = 255)
    ax[0].set_title('68 predicted face landmarks')
    ax[1].imshow(image5, cmap='gray', vmin = 1.0, vmax = 255)
    ax[1].set_title('5 predicted face landmarks')
    plt.show()