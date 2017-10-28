from utils import *
from config import *
from framework import *
from classifiers import *



training_plots = load_plots(plot_list = train_plots[1:3], plot_type= 'train')

# Give all plots an NDVI image
for p in training_plots:
    ndvi_data = ndvi_from_hs(p.images['hs'].image_data)
    p.load_image(image_type='ndvi', image_data=ndvi_data)