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






from skimage.util import view_as_windows
from itertools import combinations
class chm_graph:
    def __init__(self, i):
        self.image = i
        self.make_p_dag()
        self.make_h_dag()
        self.assign_h_dag_values()
        
        # The H dag nodes will be named the top level nodes from
        # the patch dag
        #self.h_dag_nodes = get_top_level_nodes(self.p_dag)
        #self.h_dag.add_nodes_from(self.h_dag_nodes)
    
    # if two top level p_dag nodes share any patch
    def get_shared_patches(self, node_1, node_2):
        node_1_patches = np.array(list(nx.descendants(self.p_dag, node_1)))
        node_2_patches = np.array(list(nx.descendants(self.p_dag, node_2)))
        shared = np.in1d(node_1_patches, node_2_patches)
        return(node_1_patches[shared])
        #return np.any(np.in1d(node_1_patches, node_2_patches, assume_unique=True))
    
    def fill_edges(self):
        pass
    
    # The actual image value of a p-dag node
    def node_image_value(self, node):
        return self.image[self.node_lookup_table==node][0]
    
    def node_location(self, node, with_z=False):
        loc = np.where(self.node_lookup_table==node)
        x, y = loc[0][0], loc[1][0]
        if with_z:
            z = self.node_image_value(node)
            return x,y,z
        else:
            return x,y
    
    # Should work in either 2d or 3d
    # euclidean distance of the array locations for now
    def get_node_distance(self, node1, node2, with_z=False):
        node1_loc = np.array(self.node_location(node1, with_z = with_z))
        node2_loc = np.array(self.node_location(node2, with_z = with_z))
        return np.sqrt(np.sum((node1_loc - node2_loc)**2))
        
    
    def get_top_level_nodes(self, g):
        top_level_nodes=[]
        for node in list(g.nodes):
            if len(list(g.predecessors(node)))==0:
                top_level_nodes.append(node)
        return top_level_nodes

    # DAG from a patch dag, where nodes are the full heirarchies and
    # edges are made when the heirarchies share cells. . 
    def make_h_dag(self):
        self.h_dag = nx.DiGraph()
        self.parent_nodes = self.get_top_level_nodes(self.p_dag)
        self.h_dag.add_nodes_from(self.parent_nodes)
        
        for node1, node2 in combinations(self.parent_nodes,2):
            shared_patches = self.get_shared_patches(node1, node2)
            if len(shared_patches)>0:
                if self.node_image_value(node1) > self.node_image_value(node2):
                    self.h_dag.add_edge(node1, node2)
                    self.h_dag[node1][node2]['shared_patches'] = shared_patches
                else:
                    self.h_dag.add_edge(node2, node1)
                    self.h_dag[node2][node1]['shared_patches'] = shared_patches

    def assign_h_dag_values(self):
        for node1, node2 in list(self.h_dag.edges):
            cohesion_criteria = {}
            cohesion_criteria['LD'] = self.level_depth(node1, node2)
            cohesion_criteria['SR'] = self.shared_ratio(node1, node2)
            cohesion_criteria['TD'] = self.top_distance(node1, node2)
            self.h_dag[node1][node2]['cohesion'] = cohesion_criteria

    #These are all the different cohesion criteria between h-dag nodes
    def level_depth(self, node1, node2):
        contact_node_heights=[]
        for shared_patch in self.h_dag[node1][node2]['shared_patches']:
            contact_node_heights.append(self.node_image_value(shared_patch))
        node1_min_height = np.min(self.node_image_value(node1) - contact_node_heights)
        node2_min_height = np.min(self.node_image_value(node2) - contact_node_heights)
        return 1/np.min([node1_min_height, node2_min_height])
        
    # The minimum number of cell steps to reach a contact cell
    # note sure how to do this yet...
    def node_depth(self, node1, node2):
        pass

    def shared_ratio(self, node1, node2):
        node1_total_cells = len(nx.descendants(self.p_dag, node1))
        node2_total_cells = len(nx.descendants(self.p_dag, node2))
        total_shared_cells = len(self.h_dag[node1][node2]['shared_patches'])
        return total_shared_cells / (node1_total_cells + node2_total_cells)
    
    # Horizonatl distance between parent cells
    def top_distance(self, node1, node2):
        return self.get_node_distance(node1, node2)
    
    # Directed acrylic graph from a 2d image, where children are
    # any neighbor cell with a lower value
    def make_p_dag(self):
        i = self.image.copy()
        self.num_elements = np.prod(i.shape)
        self.node_lookup_table = np.arange(self.num_elements).reshape(i.shape)
        
        self.p_dag = nx.DiGraph()
        
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
                focal_node = self.node_lookup_table[win_view_row, win_view_col]
                child_rows, child_cols = np.where(win_view[win_view_row, win_view_col] < focal_value)
                for i in range(len(child_rows)):
                    child_node = node_lookup_table[win_view_row+child_rows[i]-1, 
                                                   win_view_col+child_cols[i]-1]
                    self.p_dag.add_nodes_from([focal_node,child_node])
    
                    self.p_dag.add_edge(focal_node, child_node)
    

g = chm_graph(chm_image)







