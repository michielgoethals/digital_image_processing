import imageio
import skimage
from skimage import img_as_ubyte
import numpy as np
from scipy.spatial import Delaunay
from featureDetectionDlib import feature_detection_dlib
from piecewiseWarping import warp_image
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize

def face_morph(image1, image2, model, alphas=0.5, landmarks=False):
    # Generate features and add corners and midpoints
    pts1 = feature_detection_dlib(image1, model, corners=True)
    pts2 = feature_detection_dlib(image2, model, corners=True)
    
    # Get lists of triangles
    triangles1 = Delaunay(pts1)
    triangles2 = Delaunay(pts2)
    
    # Plot triangles if landmarks is True
    
    if landmarks:
        fig, (ax) = plt.subplots(nrows=1, ncols=2)
        ax[0].axis('off')
        ax[0].imshow(image1)
        ax[0].triplot(pts1[:,0], pts1[:,1], triangles1.simplices)
        ax[1].axis('off')
        ax[1].imshow(image2)
        ax[1].triplot(pts2[:,0], pts2[:,1], triangles2.simplices)
    
    # Create empty frames list
    frames = []
   
    # Based on alphas calculate intermediate points en morphed image
    for alpha in alphas:
        
        # Intermediate points 
        ptsm = (1-alpha)*pts1 + alpha*pts2 
        
        # Warped images
        warped1 = warp_image(img1, pts1, triangles1, ptsm, img2.shape)  # Warp image1                                
        warped2 = warp_image(img2, pts2, triangles2, ptsm, img1.shape)   # Warp image2
        
        # Morphed image
        morphed = img_as_ubyte((1-alpha)* warped1 + alpha * warped2)
    
        # Calculate new width and height of the warped images
        newWidth = int((1-alpha)*img1.shape[0] + alpha*img2.shape[0])       
        newHeight = int((1-alpha)*img1.shape[1] + alpha*img2.shape[1])
        
        # Add morphed image to the frames list
        frames.append(morphed[0:newWidth, 0:newHeight])
    
    # Return the frames list    
    return frames

def save_frames_to_video(file_name, frames):
    frame_size = frames[0].shape[:2]
    frame_rate = 10
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    writer = cv2.VideoWriter(file_name, codec, frame_rate, frame_size)

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    
    
if __name__ == "__main__":
    
    im1 = "../../imgs/faces/daenerys.jpg"       # Path to first image
    im2 = "../../imgs/faces/gal_gadot.jpg"      # Path to second image
    
    img1 = imageio.imread(im1)                  # Load the first image
    img1 = skimage.util.img_as_float(img1)      # Convert image data type

    img2 = imageio.imread(im2)                  # load the second image
    img2 = skimage.util.img_as_float(img2)      # Convert image data type
    
    # Path to model 68 landmark
    model68 = "./data/facelandmarks/shape_predictor_68_face_landmarks.dat"
    
    # Make a list of alphas from 0 to 1 in 20 steps
    alphas = np.linspace(0, 1, 50) 
    
    # Calculate frames 
    frames = face_morph(img1, img2, model68, alphas=alphas, landmarks=False)
    
    plt.subplot(121);plt.title('First frame',fontsize=30);plt.imshow(frames[0]);plt.axis('off')
    plt.subplot(122);plt.title('Last frame',fontsize=30);plt.imshow(frames[19]);plt.axis('off')                 
    
    # Save frames to MP4 file and store in same dir as this file 
    save_frames_to_video("frames_video.mp4", frames)
        
        
        