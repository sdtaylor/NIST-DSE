from skimage.external import tifffile
from skimage import io
from skimage import measure
from skimage.morphology import watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import numpy as np




def ndvi_from_hs(i):
    red_bands = np.array([51,52,53,54,55,56,57,58,59])
    nir_bands = np.array([95,96,97,98,99,100])

    i_red = i[:,:,red_bands].mean(axis=2)
    i_nir = i[:,:,nir_bands].mean(axis=2)
    ndvi = (i_nir - i_red) / (i_nir + i_red)

    return ndvi

# Returns a labeled np array and optionally coordinates of local maxima
def labels_from_watershed(height_image, ndvi_image, min_distance, ndvi_threshold,
                          return_coordinates=False):
    # Only do local maxima and watershedding where:
    height_mask = np.logical_or(ndvi_image>=ndvi_threshold, height_image>0)
    
    local_maxi = peak_local_max(height_image, 
                                 min_distance=min_distance,
                                 labels=height_mask,
                                 threshold_abs=2, indices= False)
    coordinates = peak_local_max(height_image, 
                                 min_distance=min_distance,
                                 labels=height_mask,
                                 threshold_abs=2, indices= True)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-chm_image, markers, mask=height_mask)
    
    if return_coordinates:
        return labels, coordinates
    else:
        return labels

# Draw a circle in an array of size (array_x, array_y)
# with circle center (center_x, center_y) and radius
def draw_circle(array_shape, circle_center ,radius):
    rows, cols = np.indices((array_shape[0], array_shape[1]))
    circle_center_row, circle_center_col = circle_center[0], circle_center[1]
    circle_mask = (rows - circle_center_row)**2 + (cols - circle_center_col)**2 < radius**2
    return circle_mask

# Get a single image with masks for many (potentially overlapping)
# circles. points are the center points for all circles
def get_circles_from_points(array_shape,points, radius):
    a = np.zeros(array_shape).astype(bool)
    for i in range(points.shape[0]):
        circle = draw_circle(array_shape, points[i], radius=radius)
        a = np.logical_or(a, circle)
    return a
    
#######################################################

ndvi_cutoff = 2
local_maxima_min_distance = 5
apply_max_crown_radius = True
max_crown_radius = 10

chm_image = tifffile.imread('OSBS_039_chm.tif')
hs_image = tifffile.imread('OSBS_039_hyper.tif')
ndvi = ndvi_from_hs(hs_image)

labels, coordinates = labels_from_watershed(height_image=chm_image,
                                            ndvi_image=ndvi,
                                            ndvi_threshold=ndvi_cutoff,
                                            min_distance=local_maxima_min_distance,
                                            return_coordinates=True)

if apply_max_crown_radius:
    circles = get_circles_from_points(array_shape=chm_image.shape, 
                                      points=coordinates, 
                                      radius=max_crown_radius)
    labels[~circles] = 0

plt.imshow(labels, cmap=plt.cm.spectral)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')

#fig.tight_layout()

plt.show()