import pandas as pd
import numpy as np
from shapely import wkt
import shapely
from shapely import affinity
from shapely import geometry
import fiona
import rasterio
from rasterio import plot
from rasterio import features
#from utils import *
from config import *


#Wrapper for individual images. Either RGB, P, M, etc.
class image_wrapper:
    def __init__(self, filename=None, image_data=None):
        if filename is not None:
            self.image_object = rasterio.open(filename)
            self.image_data = self.image_object.read()
            self.transform = self.image_object.transform
        elif image_data is not None:
            self.image_data = image_data
            self.transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)
        else:
            stop('need filename to read or raw image data')

        #Multiband vs single band images
        if len(self.image_data.shape)==3:
            self.bands, self.H, self.W = self.image_data.shape
        else:
            self.H, self.W = self.image_data.shape
            self.bands=1

        #Containers for class_types. polygons are a shapely
        #multipolygon object. masks are a (0,1) mask the same shape as the image
        self.class_masks={}
        self.class_polygons={}
        self.class_masks_predict={}
        self.class_polygons_predict={}

    def _polygon_to_mask(self, polygon):
        if not polygon.is_empty:
            return rasterio.features.geometry_mask(geometries = polygon,
                                                   out_shape = (self.H, self.W),
                                                   transform = self.transform,
                                                   all_touched = False,
                                                   invert = True)
        else:
            return np.zeros((self.H,self.W)).astype(bool)

    #Apply a class polygon to this image, scaling to the appropriate size
    def load_class_polygon(self, class_type, polygon):
        assert class_type != self.class_polygons, 'Class type already loaded in polygons'
        assert class_type != self.class_masks, 'Class type already loaded in masks'

        scaled_polygon = self._scale_polygon(polygon)
        self.class_polygons[class_type] = scaled_polygon
        self.class_masks[class_type] = self._polygon_to_mask(scaled_polygon)

    #Convert a mask of 0,1 to a multipolygon
    def _mask_to_polygons(self, mask):
        all_polygons=[]
        for shape, value in features.shapes(mask.astype(np.int16),
                                            mask = (mask==1),
                                            transform = self.transform):

            all_polygons.append(shapely.geometry.shape(shape))

        all_polygons = shapely.geometry.MultiPolygon(all_polygons)
        if not all_polygons.is_valid:
            all_polygons = all_polygons.buffer(0)
            #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            #need to keep it a Multi throughout
            if all_polygons.type == 'Polygon':
                all_polygons = shapely.geometry.MultiPolygon([all_polygons])
        return all_polygons

    #Make a new class based on a mask. Scaling to image size if neccessary
    #Note this and load_prediction() only differ wher ethey are stored. 
    #A new class should be considered a "true" mask. while the prediction is
    #the prodcut of a model. 
    #Nevermind, these should be combined. mabye.
    def load_class_mask(self, class_type, class_mask, reshape=True):
        #If the prediction is coming from model output and is a 1d array
        assert np.any(np.unique(class_mask) == [0,1]), 'Prediction array not just 1 and 0'
        if reshape:
            try:
                class_mask = np.reshape(class_mask, (self.H, self.W))
            except:
                print('Could not reshape prediction from ', class_mask.shape, ' to ', (self.H, self.W))

        self.class_masks[class_type] = class_mask
        self.class_polygons[class_type] = self._mask_to_polygons(class_mask)

    #Load in a class prediction. Can be either polygon or mask. A mask
    #will be converted to polygon. 
    def load_prediction(self, class_type, predict, is_mask=True, reshape=True):
        if is_mask:
            assert np.any(np.unique(predict) == [0,1]), 'Prediction array not just 1 and 0'
            #Reshape needed if the prediction is coming model output and is a 1d array
            if reshape:
                try:
                    predict = np.reshape(predict, (self.H, self.W))
                except:
                    print('Could not reshape prediction from ', predict.shape, ' to ', (self.H, self.W))
            self.class_masks_predict[class_type] = predict
            predict = self._mask_to_polygons(predict)

        self.class_polygons_predict[class_type] = predict

    #Return a class polygon prediction
    def get_polygon_prediction(self, class_type):
        assert class_type in self.class_polygons_predict, 'Class type not in prediction polygons'
        return self._scale_polygon(self.class_polygons_predict[class_type], to_image_coord=False)

    #Return a class polygon
    def get_polygon(self, class_type):
        assert class_type in self.class_polygons, 'Class type not polygons'
        return self._scale_polygon(self.class_polygons[class_type], to_image_coord=False)

    #Returns a num_pixel x num_band array of values falling within this class type
    def get_class_values(self, class_type):
        class_mask = self.class_masks[class_type]
        #Empty mask means this class is not in this image
        if np.sum(class_mask)==0:
            return None

        for band in range(self.bands):
            band_data = self.image_data[band][np.where(class_mask)]
            if band == 0:
                class_data = band_data
            else:
                class_data = np.vstack((class_data, band_data))

        return class_data.transpose()

    #Return num_pixel x num_band array of randomly located pixels.
    #If mask is included, exclude from consideration pixels where mask is true.
    #This is to train classifiers without using every pixel in an image,
    #the mask will be the class that is being trained on.
    def get_random_pixels(self, num_pixels, class_mask):
        #load the mask for this class if a number is specified instead of actual mask
        #class_mask is None if the class isn't present in this plot
        if type(class_mask) is int:
            class_mask = self.class_masks[class_mask]
            if class_mask is None:
                class_mask = np.zeros((self.H, self.W))

        #Pixels that are excluded from consideration (because they are of class class_mask)
        #are false in include_mask
        include_mask = np.invert(class_mask.astype(bool))

        #If the number of random pixels asked for is more than what is available (in small images), 
        #send back all pixels not in the class mask
        if self.H * self.W <= num_pixels:
            random_pixels=np.ones(self.H * self.W)
        else:
            random_pixels=np.hstack((np.ones(num_pixels),
                                     np.zeros((self.H*self.W)-num_pixels)))
            np.random.shuffle(random_pixels)

        random_pixels = random_pixels.reshape((self.H, self.W)).astype(bool)
        random_pixels = (random_pixels & include_mask)

        #the full amount of pixels requested won't be returned sometimes.
        #there is probably a way to return exactly num_pixels while still incorporating
        #class_mask, but it's not a big deal
        print('Pixels requested, pixels returned')
        print(num_pixels, np.sum(random_pixels))

        for band in range(self.bands):
            band_data = self.image_data[band][np.where(random_pixels)]
            if band == 0:
                random_data = band_data
            else:
                random_data = np.vstack((random_data, band_data))

        return random_data.transpose()

    #Return a num_pixel x num_band array of all values.
    def get_all_values(self):
        for band in range(self.bands):
            band_data = self.image_data[band].reshape(self.W*self.H)
            if band == 0:
                all_data = band_data
            else:
                all_data = np.vstack((all_data, band_data))

        return all_data.transpose()

# For shapefile data, either either a .shp filename, a
# multipolygon object, or an image data (as a 0,1 mask)
# which will be converted to a multipolygon
class shapefile_wrapper:
    def __init__(self, filename=None, multipolygon=None, image_mask=None):
        #Load the shapefile into a multipolygon
        if filename is not None:
            self.filename=filename
            self.file_object = fiona.open(filename)
            self._shapefile_to_multipolygon()
        elif multipolygon is not None:
            # should be a MultiPolygon object
            pass
        elif image_mask is not None:
            self.image_mask = image_mask
            self.multipolygon = 
            stop('Need filename or shapefile data')
            
    def _shapefile_to_multipolygon(self):
        polygons = []
        for p in self.file_object:
            polygons.append(shapely.geometry.shape(p['geometry']))
        self.multipolygon = shapely.geometry.MultiPolygon(polygons)

    #Convert a mask of 0,1 to a multipolygon
    def _mask_to_polygons(self, mask):
        all_polygons=[]
        for shape, value in features.shapes(mask.astype(np.int16),
                                            mask = (mask==1),
                                            transform = self.transform):

            all_polygons.append(shapely.geometry.shape(shape))

        all_polygons = shapely.geometry.MultiPolygon(all_polygons)
        if not all_polygons.is_valid:
            all_polygons = all_polygons.buffer(0)
            #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            #need to keep it a Multi throughout
            if all_polygons.type == 'Polygon':
                all_polygons = shapely.geometry.MultiPolygon([all_polygons])
        return all_polygons

    def _multipolygon_to_shapefile(self):
        pass
    def write_shapefile(self, new_filename):
        pass



#plots includes include all image types and polygon info
class plot_wrapper:
    def __init__(self, plot_id):
        self.plot_id = plot_id
        self.images={}

        self.polygons = {}
        #self.polygons_predict = {}

    def load_train_polygons(self, filename):
        self.train_polygons=shapefile_wrapper(filename=filename)

    def load_image(self, image_type, image_filename=None, image_data=None):
        self.images[image_type] = image_wrapper(filename=image_filename, image_data=image_data)

    def load_predicted_polygons(self, polygons):
        self.polygons_predict = polygons
        
    #Pull class from an image and load it as a prediction
    def pull_prediction(self, class_type, image):
        assert class_type not in self.polygons_predict, 'Class prediction type already exists in plot'
        self.polygons_predict[class_type] = self.images[image].get_polygon_prediction(class_type)

    #Pull class from one or all of the images
    def pull_class(self, class_type, image):
        assert class_type not in self.polygons, 'Class type already exists in plot'
        self.polygons[class_type] = self.images[image].get_polygon(class_type)

    #Apply a new class type to the image
    def push_class(self, class_type, image):
        self.images[image].load_class_polygon(class_type, self.polygons[class_type])

    #Calculate and store all scores for class predictions
    def calculate_class_scores(self):
        self.class_scores={}
        for class_type in range(1,11):
            this_class_scores={}
            if class_type not in self.polygons_predict:
                continue
            this_class_scores['tp'] = self.polygons[class_type].intersection(self.polygons_predict[class_type]).area
            this_class_scores['fp'] = self.polygons_predict[class_type].area - this_class_scores['tp']
            this_class_scores['fn'] = self.polygons[class_type].area - this_class_scores['tp']
            jaccard_denominator = np.sum([this_class_scores[m] for m in ['tp','fp','fn']])
            this_class_scores['jaccard']  = this_class_scores['tp'] / jaccard_denominator if jaccard_denominator > 0 else 0
            self.class_scores[class_type]=this_class_scores

    #return true positive, false positive, false negative, jaccard score
    def get_class_score(self, class_type):
        assert self.plot_type != 'test', 'plot is training, cannot calculate score'
        #assert class_type in self.polygons_predict, 'Class type not in predictions'
        if class_type not in self.polygons_predict:
            return None

        tp = self.class_scores[class_type]['tp']
        fp = self.class_scores[class_type]['fp']
        fn = self.class_scores[class_type]['fn']
        jaccard = self.class_scores[class_type]['jaccard']
        return(tp,fp,fn,jaccard)

    #Get a list of dicts of predictions for submission
    def get_wkt_prediction(self):
        all_class_predictions=[]
        for class_type in range(1,11):
            if self.polygons_predict[class_type].type == 'GeometryCollection':
                wkt_predicted = 'MULTIPOLYGON EMPTY'
            else:
                wkt_predicted = self.polygons_predict[class_type].wkt

            all_class_predictions.append({'ImageId':self.plot_id,
                                         'ClassType':class_type,
                                         'MultipolygonWKT':wkt_predicted})
        return all_class_predictions

    #Apply any new classes to the images
    def update_all_image_polygons(self):
        for image_type, image_object in self.images.items():
            for class_type, polygon in self.polygons.items():
                if class_type not in image_object.class_polygons:
                    image_object.load_class_polygon(class_type, polygon)


#Load a list of testing plots
def load_plots(plot_list, plot_type, image_types=image_types_to_load):
    all_plot_data=[]
    for i, plot_id in enumerate(plot_list):
        plot_data = plot_wrapper(plot_id)
        plot_data.plot_type=plot_type

        for image_type in image_types:
            if image_type=='hs':
                image_path = hs_image_dir + plot_id+'_hyper.tif'
            elif image_type=='chm':
                image_path = chm_image_dir + plot_id+'_chm.tif'

            plot_data.load_image(image_type, image_path)

        # Load canopy shapefiles for training data
        if plot_type=='train':
            shapefile_path = training_polygons_dir+'ITC_'+plot_id+'.shp'
            plot_data.load_train_polygons(shapefile_path)
        
        all_plot_data.append(plot_data)

    return all_plot_data
