from skimage.external import tifffile
from scipy.ndimage import center_of_mass
import numpy as np
import networkx as nx

chm_image = tifffile.imread('OSBS_006_chm.tif')

# Use a 10x10 plot for testing
#chm_image = chm_image[0:30,0:30]

from skimage.util import view_as_windows
from itertools import combinations
class chm_graph:
    def __init__(self, i):
        self.image = i
        
        # The initial dag from all gridded cell values
        self.make_cell_dag()
        
        # The heiarchichal dag
        self.make_h_dag()
        # The cohesion values
        self.assign_h_dag_values()
   
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
    
    # will work in either 2d or 3d
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

    ##############################################
    # Functions for for bulding the H-DAG
    ##############################################
    # DAG from a patch dag, where nodes are the full heirarchies and
    # edges are made when the heirarchies share cells. . 
    def make_h_dag(self):
        self.h_dag = nx.DiGraph()
        self.parent_nodes = self.get_top_level_nodes(self.cell_dag)
        self.h_dag.add_nodes_from(self.parent_nodes)
        
        for p_node1, p_node2 in combinations(self.parent_nodes,2):
            shared_patches = self.get_shared_patches(p_node1, p_node2)
            if len(shared_patches)>0:
                if self.node_image_value(p_node1) > self.node_image_value(p_node2):
                    self.h_dag.add_edge(p_node1, p_node2)
                    self.h_dag[p_node1][p_node2]['shared_patches'] = shared_patches
                else:
                    self.h_dag.add_edge(p_node2, p_node1)
                    self.h_dag[p_node2][p_node1]['shared_patches'] = shared_patches

    # Shared cells between two parent heiarchies.
    def get_shared_patches(self, p_node1, p_node2):
        p_node1_patches = np.array(list(nx.descendants(self.cell_dag, p_node1)))
        p_node2_patches = np.array(list(nx.descendants(self.cell_dag, p_node2)))
        shared = np.in1d(p_node1_patches, p_node2_patches)
        return(p_node1_patches[shared])

    def assign_h_dag_values(self):
        for p_node1, p_node2 in list(self.h_dag.edges):
            cohesion_criteria = {}
            cohesion_criteria['LD'] = self.level_depth(p_node1, p_node2)
            cohesion_criteria['SR'] = self.shared_ratio(p_node1, p_node2)
            cohesion_criteria['TD'] = self.top_distance(p_node1, p_node2)
            self.h_dag[p_node1][p_node2]['cohesion'] = cohesion_criteria

    ##############################################
    # Predicting each original cell in the imaged based on segmentation
    ##############################################

    def get_image_predict(self):
        # A list to store parent ID's (now known as trees) for each original cell. 
        # cells can belong to more than 1, which is resolved in 2 steps
        for cell_node in list(self.cell_dag.nodes):
            self.cell_dag.nodes[cell_node]['tree_id']=[]
                
        # Assign each cell to it's tree id's. The node id's from the final top
        # level h_dag are now considered the top of the trees. 
        # Also get the center of each tree to resolve potential conflicts
        for tree_id in self.get_top_level_nodes(self.h_dag_predict):
            self.cell_dag.nodes[tree_id]['tree_id'].append(tree_id)
            self.h_dag_predict.nodes[tree_id]['tree_center']=self.get_tree_center(tree_id)
            for cell_node in self.get_all_tree_cells(tree_id):
                self.cell_dag.nodes[cell_node]['tree_id'].append(tree_id)
                
        # resolve multiple tree IDs
        for cell_node in list(self.cell_dag.nodes):
            tree_ids = self.cell_dag.nodes[cell_node]['tree_id']
            if len(tree_ids)==0:
                raise Exception('cell node has no tree ids: '+str(cell_node))
            elif len(tree_ids)>1:
                self.cell_dag.nodes[cell_node]['tree_id'] = self.get_nearest_tree(cell_node, tree_ids)
            else:
                # If everything is sound make it a single id instead of a list
                self.cell_dag.nodes[cell_node]['tree_id']=tree_ids[0]
        
    # The nearest tree, based on tree center, to a cell node
    def get_nearest_tree(self, cell_node, tree_ids):
        tree_locations = [self.h_dag_predict.nodes[tree]['tree_center'] for tree in tree_ids]
        cell_location = np.array(self.node_location(cell_node))
        tree_distances = [np.sqrt(np.sum((tree_loc - cell_location)**2)) for tree_loc in tree_locations]
        tree_distances, tree_ids = zip(*sorted(zip(tree_distances, tree_ids)))
        return tree_ids[0]

    def get_tree_center(self, tree_id):
        all_tree_cells = np.array(list(nx.descendants(self.cell_dag, tree_id)))
        tree_mask = np.in1d(self.node_lookup_table, all_tree_cells).reshape(self.image.shape)
        center = np.array(center_of_mass(tree_mask))
        return center
    
    # Return *all* cells of a tree, which can include multiple heiarchies
    def get_all_tree_cells(self, tree_id):
        all_cells=[]
        # Cells directly under this tree
        all_cells.extend(list(nx.descendants(self.cell_dag, tree_id)))
        
        # Cells under the connecting sub heiarchies
        for sub_tree in list(nx.descendants(self.h_dag_predict, tree_id)):
            all_cells.append(sub_tree)
            all_cells.extend(list(nx.descendants(self.cell_dag, sub_tree)))
        
        return all_cells
    
    ##############################################
    # Functions for doing the segmentation of trees in the H-DAG
    ##############################################
    def apply_segmentation(self, wt, weights):
        self.create_prediction_h_dag()
        self.apply_edge_weights(**weights)
        self.cut_weak_edges(wt)

    def apply_edge_weights(self, **weights):
        for p_node1, p_node2 in list(self.h_dag_predict.edges):
            cohesion = self.h_dag[p_node1][p_node2]['cohesion']
            edge_weight = 0
            for cohesion_param, cohesion_value in cohesion.items():
                edge_weight += cohesion_value * weights[cohesion_param]
            self.h_dag_predict[p_node1][p_node2]['WE']=edge_weight

    def cut_weak_edges(self, wt):
        for p_node1, p_node2 in list(self.h_dag_predict.edges):
            if self.h_dag_predict[p_node1][p_node2]['WE'] < wt:
                self.h_dag_predict.remove_edge(p_node1, p_node2)
        
        # Also ensure no parent node has > 1 parent itself
        # the non-maximal inbound edge part of the paper
        for p_node in list(self.h_dag_predict.nodes):
            parents_of_p_node = list(self.h_dag_predict.predecessors(p_node))
            if len(parents_of_p_node)>1:
                # Cut all edges except the one with highest weight
                for p_p_node in self.get_sorted_parents(p_node)[:-1]:
                    self.h_dag_predict.remove_edge(p_p_node, p_node)

    # Get the weights of higher connected parent nodes in the H_DAG
    # only used to cut extra edges in the prediction h_dag
    def get_sorted_parents(self, p_node):
        weights=[]
        parents_of_p_node = list(self.h_dag_predict.predecessors(p_node))
        for p_p_node in parents_of_p_node:
            weights.append(self.h_dag_predict[p_p_node][p_node]['WE'])
            
        weights, parents_of_p_node = zip(*sorted(zip(weights, parents_of_p_node)))
        return parents_of_p_node

    # Fitting will require manipulating the h-dag over and over from scratch
    # so only do it on copies
    def create_prediction_h_dag(self):
        self.h_dag_predict = self.h_dag.copy()
        
    ##############################################
    # Functions for cohesion criteria
    ##############################################
    def level_depth(self, p_node1, p_node2):
        contact_node_heights=[]
        for shared_patch in self.h_dag[p_node1][p_node2]['shared_patches']:
            contact_node_heights.append(self.node_image_value(shared_patch))
        p_node1_min_height = np.min(self.node_image_value(p_node1) - contact_node_heights)
        p_node2_min_height = np.min(self.node_image_value(p_node2) - contact_node_heights)
        return 1/np.min([p_node1_min_height, p_node2_min_height])
        
    # The minimum number of cell steps to reach a contact cell
    # note sure how to do this yet.
    # just the horizontal distance between parent cells and contact cells
    def node_depth(self, p_node1, p_node2):
        pass

    def shared_ratio(self, p_node1, p_node2):
        p_node1_total_cells = len(nx.descendants(self.cell_dag, p_node1))
        p_node2_total_cells = len(nx.descendants(self.cell_dag, p_node2))
        total_shared_cells = len(self.h_dag[p_node1][p_node2]['shared_patches'])
        return total_shared_cells / (p_node1_total_cells + p_node2_total_cells)
    
    # Horizonatl distance between parent cells
    def top_distance(self, p_node1, p_node2):
        return self.get_node_distance(p_node1, p_node2)
        
    ##############################################
    # Functions for building initial dag based off individual
    # raster cells.
    ##############################################
    
    # Directed acrylic graph from a 2d image, where children are
    # any neighbor cell with a lower value
    def make_cell_dag(self):
        i = self.image.copy()
        self.num_elements = np.prod(i.shape)
        self.node_lookup_table = np.arange(self.num_elements).reshape(i.shape)
        
        self.cell_dag = nx.DiGraph()
        
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
                    self.cell_dag.add_nodes_from([focal_node,child_node])

                    self.cell_dag.add_edge(focal_node, child_node)
    

g = chm_graph(chm_image)







