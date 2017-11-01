from skimage.external import tifffile
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from scipy.sparse import csgraph
import numpy as np


chm_image = tifffile.imread('OSBS_006_chm.tif')

def levels_to_3d(a, H, W):
    levels_3d = np.expand_dims(a, axis=0)
    levels_3d = np.expand_dims(levels_3d, axis=0)
    levels_3d = np.repeat(levels_3d, W, 1)
    levels_3d = np.repeat(levels_3d, H, 0)
    return levels_3d

def image_to_3d(i, num_levels):
    i = np.expand_dims(i, axis=2)
    return np.repeat(i, num_levels, axis=2)

horizontal_levels = np.array(range(1,23,2))
level_lower_limit = levels_to_3d(horizontal_levels - 1, 80,80)
level_upper_limit = levels_to_3d(horizontal_levels + 1, 80,80)

# A (H, W, h_levels) image of zeros
chm_3d = image_to_3d(chm_image, num_levels=len(horizontal_levels))

chm_3d = np.logical_and(chm_3d >= level_lower_limit , chm_3d < level_upper_limit)



# =============================================================================
# do 3x3 smoother
# for every element (starts with element 1,1)(parent):
#     for every adjacent element that is smaller (child):
#         add to graph (parent -> child)
# redo this by hand  for edges.
# =============================================================================


num_elements = np.prod(chm_image.shape)
node_lookup_table = np.arange(num_elements).reshape(chm_image.shape)
dense_graph = np.zeros((num_elements, num_elements))

smoother = view_as_windows(chm_image, (3,3))
for smoother_row in range(smoother.shape[0]):
    for smoother_col in range(smoother.shape[1]):
        focal_value = smoother[smoother_row, smoother_col][1,1]
        focal_node = node_lookup_table[smoother_row+1, smoother_col+1]
        child_row, child_col = np.where(smoother[smoother_row, smoother_col] < focal_value)
        for i in range(len(child_row)):
            child_node = node_lookup_table[smoother_row+child_row[i], 
                                           smoother_col+child_col[i]]
            dense_graph[focal_node,child_node]=1
            #dense_graph[child_node,focal_node]=1

# Redo for the edges
# ... someday






