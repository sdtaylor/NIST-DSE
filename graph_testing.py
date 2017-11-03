################################################
import matplotlib.pyplot as plt

import config
import utils
import classifiers
from graph_classifier import graph_classifier
import framework as F

training_plots = F.load_plots(plot_list = config.train_plots[1:2], plot_type= 'train')
training_plots[0].load_chm_graph()

model = graph_classifier()

parameters = {'LD':2/10,'SR':5/10,'TD':3/10, 'wt':5}