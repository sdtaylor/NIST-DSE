from skimage.external import tifffile
import numpy as np
from graph_framework import chm_graph

################################################
import matplotlib.pyplot as plt

                    
chm_image = tifffile.imread('OSBS_032_chm.tif')

# Use a 10x10 plot for testing
#chm_image = chm_image[0:30,0:30]
g = chm_graph(chm_image)

weights = {'LD':2/10,'SR':5/10,'TD':3/10}
weight_threshold = 5

g.apply_segmentation(weight_threshold, weights)