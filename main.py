import config
import utils
import classifiers
import framework as F

training_plots = F.load_plots(plot_list = config.train_plots, plot_type= 'train')
testing_plots = F.load_plots(plot_list = config.test_plots, plot_type= 'test')


# Give all plots an NDVI image
for p in training_plots:
    ndvi_data = utils.ndvi_from_hs(p.images['hs'].image_data)
    p.load_image(image_type='ndvi', image_data=ndvi_data)
for p in testing_plots:
    ndvi_data = utils.ndvi_from_hs(p.images['hs'].image_data)
    p.load_image(image_type='ndvi', image_data=ndvi_data)

# some predifined  parameters for testing
#parameters = {'maxima_min_distance':9,
#              'ndvi_threshold':0.9,
#              'max_crown_radius':6}

#model = watershed_classifier(parameters=parameters)
model = classifiers.watershed_classifier()

model.fit(training_plots, verbose=True)

testing_plots = model.predict(testing_plots)

for p in testing_plots:
    filename = config.prediction_polygons_dir+'itc_subm_'+p.plot_id+'.shp'
    p.write_prediction('canopy', new_filename=filename)
