from skimage.external import tifffile
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
import numpy as np
import networkx as nx

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

# Use a 10x10 plot for testing
#chm_image = chm_image[0:5,0:5]

num_elements = np.prod(chm_image.shape)
node_lookup_table = np.arange(num_elements).reshape(chm_image.shape)

h_dag = nx.DiGraph()
h_dag.add_nodes_from(range(num_elements))

# Directed acrylic graph from a 2d image, where children are
# any neighbor cell with a lower value
from skimage.util import view_as_windows
def make_dag(i):
    num_elements = np.prod(i.shape)
    node_lookup_table = np.arange(num_elements).reshape(i.shape)
    
    h_dag = nx.DiGraph()
    #h_dag.add_nodes_from(range(num_elements))
    
    # Add a buffer so that a 3x3 moving window will work on edges
    # np.inf ensures the buffer is never considered in dag
    i = np.pad(i, (1,1), mode='constant', constant_values=np.inf)
    
    # Set the ground to infinity so they are not counted as children
    i[i==0] = np.inf
    
    #This will iterate over every cell and consider the 8 neighbors
    win_view = view_as_windows(i, (3,3))
    for win_view_row in range(win_view.shape[0]):
        for win_view_col in range(win_view.shape[1]):
            focal_value = win_view[win_view_row, win_view_col][1,1]
            # Cannot be parent if it's ground
            if np.isinf(focal_value):
                continue
            focal_node = node_lookup_table[win_view_row, win_view_col]
            child_rows, child_cols = np.where(win_view[win_view_row, win_view_col] < focal_value)
            for i in range(len(child_rows)):
                child_node = node_lookup_table[win_view_row+child_rows[i]-1, 
                                               win_view_col+child_cols[i]-1]
                h_dag.add_nodes_from([focal_node,child_node])

                h_dag.add_edge(focal_node, child_node)
    
    return h_dag


def get_top_level_nodes(di_graph):
    top_level_nodes=[]
    for node in list(di_graph.nodes):
        if len(list(di_graph.predecessors(node)))==0:
            top_level_nodes.append(node)
    return top_level_nodes


h_dag = make_dag(chm_image)







