
from piecewiseWarping import warp_image
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from featureDetectionDlib import feature_detection_dlib
from pyramidBlending import get_laplacian_pyramid, get_gaussian_pyramid, pyramid_blend,plot_pyramid
import matplotlib.pyplot as plt
from skimage.draw import polygon
import imageio
import cv2
import os


def convexHull(img,model,DEBUG=False, limited=None):
	"""
	Calculates the convex hull of the face of an image

	Args:
		img : Image to fit convex hull to
		model : Model of the featurepoints of the face :  shape_predictor_68_face_landmarks.dat
		DEBUG: plot face with convex hull and triangles

	Returns:
		convex_h: Points that make up the convex hull
		traingles: The Delauney triangles from the convexhull points
	"""
	pts = feature_detection_dlib(img,model)
	convex = ConvexHull(pts[:-8])
	convex_h = np.zeros((0,2), tuple)
	if limited is None:
		for i in convex.vertices:
			convex_h = np.vstack((convex_h,pts[i]))
	if limited is not None:
		for i in limited:
			convex_h = np.vstack((convex_h,pts[i]))
	convex_b = np.vstack((convex_h, pts[1]))                        # convexHull samenplaatsen met cornerpunten om Ddelaunay triangles te kunnen maken     
	triangles = Delaunay(convex_b)
	if DEBUG:
		fig, (ax) = plt.subplots(nrows=1, ncols=1)
		ax.imshow(img)
		ax.triplot(convex_b[:,0], convex_b[:,1], triangles.simplices)
		plt.show()
	return convex_h, triangles, convex_b, convex.vertices


def create_mask(shape, pts):
	"""
	Make the mask of the convex hull

	Args:
		shape : Shape of warped image
		pts : Points of the convex hull

	Returns:
		Mask: Return the mask for the face
	"""
	mask = np.zeros(shape, dtype=np.uint8)
	row = pts[:,1]
	col = pts[:,0]
	rr, cc = polygon(row, col)
	mask[rr, cc] = 1
	return mask

if __name__ == "__main__":
	image_folder = "././imgs/faces/"
	img_name = "meghan_markle.jpg"
	img_name2 = "queen.jpg"
	img1 = imageio.imread(image_folder+img_name)
	img2 = imageio.imread(image_folder+img_name2)

	pwd = os.path.dirname(__file__)
	model = pwd + "/data/facelandmarks/shape_predictor_68_face_landmarks.dat"
	pts1 = feature_detection_dlib(img1 ,model, True)
	pts2 = feature_detection_dlib(img2 ,model, True)
	triangles2 = Delaunay(pts2)

	warped2 = warp_image(img2, pts2, triangles2, pts1, img1.shape)
	#plt.imshow(warped2)
	#plt.title("Warped images")
	#plt.show()
	con1,tri1,conf,vert = convexHull(warped2,model=model)
	mask = create_mask(warped2.shape, con1)
	#plt.imshow(mask*255)
	#plt.title("Convex hull mask")
	#plt.show()
	r,c = img1.shape[:2]
	mask2 = mask[:r,:c]
	#plt.imshow(mask2)
	#plt.title("Convex hull mask resized")
	#plt.show()
	queen = warped2[:r,:c]
	alpha = img1/255*(1-mask2) + queen*mask2
	plt.imshow(alpha)
	plt.title("Alpha blending")
	plt.show()


	
	# Test pyramid blend function
	gmask = get_gaussian_pyramid(mask2*255)
	Limg1 = get_laplacian_pyramid(img1)
	Limg2 = get_laplacian_pyramid(queen*255)

	blended = pyramid_blend(gmask ,Limg1, Limg2)

	plt.imshow(blended)
	plt.title("Pyramid blending")
	plt.show()






