from utils import *
from config import *
from framework import *
from classifiers import *



training_plots = load_plots(plot_list = train_plots, plot_type= 'train')
testing_plots = load_plots(plot_list = test_plots, plot_type= 'test')


# Give all plots an NDVI image
for p in training_plots:
    ndvi_data = ndvi_from_hs(p.images['hs'].image_data)
    p.load_image(image_type='ndvi', image_data=ndvi_data)
for p in testing_plots:
    ndvi_data = ndvi_from_hs(p.images['hs'].image_data)
    p.load_image(image_type='ndvi', image_data=ndvi_data)

# some predifined  parameters for testing
parameters = {'maxima_min_distance':9,
              'ndvi_threshold':0.9,
              'max_crown_radius':6}

#model = watershed_classifier(parameters=parameters)
model = watershed_classifier()

model.fit(training_plots)

testing_plots = model.predict(testing_plots)

for p in testing_plots:
    filename = prediction_polygons_dir+'itc_subm_'+p.plot_id+'.shp'
    p.write_prediction('canopy', new_filename=filename)