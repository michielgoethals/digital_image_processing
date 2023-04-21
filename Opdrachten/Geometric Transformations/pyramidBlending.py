import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce, pyramid_expand
import cv2
from packaging import version
import skimage

def plot_pyramid(pyramid):
    """
    Create and plot a stitched image of a pyramid

    Args:
        pyramid : Pyramid array to be plotted
    """
#     if(pyramid[0].shape[0] < pyramid[len(pyramid)-1].shape[0]):
#         pyramid = pyramid[::-1]
    rows, cols, dim = pyramid[0].shape
    composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
    composite_image[:rows, :cols, :] = pyramid[0]

    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    fig, ax = plt.subplots()

    ax.imshow(composite_image)
    plt.show()


def get_gaussian_pyramid(img,downscale=2,**kwargs):
    """
    Get the gaussian pyramid of an image

    Args:
        img : Image to make a laplacian pyramid off.
        Downscale (int, optional): Downscale value. Defaults to 2.

    Returns:
        G_pyramid: Array of gaussian pyramid images
    """
    row = img.shape[0]
    column = img.shape[1]
    G_pyramid = []
    var = img/255

    while row > 8 and column > 8:
        G_pyramid.append(var)
        if version.parse(skimage.__version__) < version.parse("0.19.0"):       # elke foto toegevoegd aan ims array
            var = pyramid_reduce(var, downscale=downscale, multichannel=True)                              
        else:  
            var = pyramid_reduce(var, downscale=downscale, channel_axis=2)
        row = var.shape[0]
        column = var.shape[1]
    G_pyramid.append(var)
    return G_pyramid


def get_laplacian_pyramid(img,upscale=2, **kwargs):
    """
    Get the laplacian pyramid from an image's gaussian pyramid or the image itself

    Args:
        img : Image to make a laplacian pyramid off.
        upscale (int, optional): Upscale value for gaussian pyramid. Defaults to 2.

    Returns:
        L_pyramid: Array of laplacian pyramid images
    """
    pyramid = get_gaussian_pyramid(img,upscale=upscale)
    L_pyramid = []
    for i in range(len(pyramid)-1,-1,-1):
        if i == len(pyramid)-1:
            L_pyramid.append(pyramid[i])
        else:
            if version.parse(skimage.__version__) < version.parse("0.19.0"):       # elke foto toegevoegd aan ims array
                 prev = pyramid_expand(pyramid[i+1],upscale=upscale, multichannel=True)                             
            else:  
                prev = pyramid_expand(pyramid[i+1],upscale=upscale, channel_axis=2)
            L_pyramid.append(cv2.resize(pyramid[i],dsize=(prev.shape[1],prev.shape[0]),interpolation=cv2.INTER_CUBIC)-prev)
    L_pyramid=L_pyramid[::-1]
    return L_pyramid

def reconstruct_image_from_laplacian_pyramid(pyramid):
    """
    Reconstruct an image from its laplacian pyramid

    Args:
        pyramid : Laplacian pyramid to be reconstructed

    Returns:
        R[len(R)-1]: The reconstructed image from the laplacian pyramid
        
    """
    R = []
    #loop through the pyramid from the bottom to the top
    for i in range(len(pyramid)-1,-1,-1):
        if i == len(pyramid)-1:
            R.append(pyramid[len(pyramid)-1])
        else:
            #Rescale -> vanaf versie skimage v0.19 is multichannel veranderd naar channel_axis
            if version.parse(skimage.__version__) < version.parse("0.19.0"):       # elke foto toegevoegd aan ims array
                 prev = pyramid_expand(R[len(pyramid)-2-i], multichannel=True)                              
            else:  
                prev = pyramid_expand(R[len(pyramid)-2-i], channel_axis=2)
            R.append(cv2.resize(pyramid[i],dsize=(prev.shape[1],prev.shape[0]),interpolation=cv2.INTER_CUBIC)+prev)
    return R[len(R)-1]




if __name__ == "__main__":
    image_folder = "../../imgs/faces/"
    img_name = "superman.jpg"
    img = imageio.imread(image_folder+img_name)
    plt.figure()
    plt.imshow(img)
    plt.show()

    plot_pyramid(get_gaussian_pyramid(img))
    plot_pyramid(get_laplacian_pyramid(img))
    plt.imshow(reconstruct_image_from_laplacian_pyramid(get_laplacian_pyramid(img)))
    plt.show()