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
import config

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
            
            # Don't store the band dimension if it's only a single band
            if self.bands==1:
                self.image_data = self.image_data[0]
        else:
            self.H, self.W = self.image_data.shape
            self.bands=1.
            
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
    def __init__(self, filename=None, multipolygon=None, image_mask=None,
                 transform=None):
        #Load the shapefile into a multipolygon
        if filename is not None:
            self.filename=filename
            self.file_object = fiona.open(filename)
            self._shapefile_to_multipolygon()
        elif multipolygon is not None:
            # should be a MultiPolygon object
            pass
        elif image_mask is not None:
            assert transform is not None,'transform must be set to load mask'
            self.transform = transform
            self.image_mask = image_mask
            self.multipolygon = self._mask_to_multipolygon(image_mask)
        else:
            stop('Need filename or shapefile data')
            
    def _shapefile_to_multipolygon(self):
        polygons = []
        for p in self.file_object:
            polygons.append(shapely.geometry.shape(p['geometry']))
        self.multipolygon = shapely.geometry.MultiPolygon(polygons)
        
        if not self.multipolygon.is_valid:
            self.multipolygon = self.multipolygon.buffer(0)
    #Convert a mask of 0,1 to a multipolygon
    def _mask_to_multipolygon(self, mask):
        all_polygons=[]
        for shape, value in features.shapes(mask.astype(np.int16),
                                            mask = (mask==1),
                                            transform = self.transform):

            all_polygons.append(shapely.geometry.shape(shape))

        all_polygons = shapely.geometry.MultiPolygon(all_polygons)
        if not all_polygons.is_valid:
            all_polygons = all_polygons.buffer(0)
            #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            #need to keep it a Multipolygon throughout
            if all_polygons.type == 'Polygon':
                all_polygons = shapely.geometry.MultiPolygon([all_polygons])
        return all_polygons

    def _multipolygon_to_shapefile(self):
        pass
    def write_shapefile(self, new_filename, plot_id):
        # Write the multipolygon object as a shapefile
        schema = {'geometry':'Polygon',
                   'properties':{'Plot_ID':'str',
                                 'crown_id':'int'}}
        with fiona.collection(new_filename, 'w',
                              'ESRI Shapefile', schema) as output:
            for crown_id, polygon in enumerate(self.multipolygon,1):
                output.write({'properties': {'Plot_ID':plot_id,
                                             'crown_id':crown_id},
                              'geometry':shapely.geometry.mapping(polygon)
                            })



#plots includes include all image types and polygon info
class plot_wrapper:
    def __init__(self, plot_id):
        self.plot_id = plot_id
        self.images={}

        self.polygons = {}
        self.polygons_predict = {}

    def load_train_polygons(self, class_type, filename):
        self.polygons[class_type]=shapefile_wrapper(filename=filename)

    def load_image(self, image_type, image_filename=None, image_data=None):
        self.images[image_type] = image_wrapper(filename=image_filename, image_data=image_data)

    # Set this images transform info from either a loaded image
    # or directly
    def set_transform(self, image_type=None, transform=None):
        if image_type is not None:
            assert image_type in self.images, 'Cannot set transform, image not loaded: '+image_type
            self.transform = self.images[image_type].transform
        elif transform is not None:
            self.transform = transform
        else:
            print('Transform not set')
        
    # Load a mask of 0,1 where 1 is the predicted canopy area
    # This will convert it to a multipolygon object
    def load_prediction_mask(self, class_type, mask):
        self.polygons_predict[class_type] = shapefile_wrapper(image_mask=mask,
                                                              transform=self.transform)
        
    def write_prediction(self, class_type, new_filename):
        assert class_type in self.polygons_predict, class_type+' not in polygons_predict'
        self.polygons_predict[class_type].write_shapefile(new_filename=new_filename,
                                                          plot_id = self.plot_id)
        
    # Calculate jaccard error for a given class beween polygons and polygons_predict
    def get_jaccard_error(self, class_type):
        assert class_type in self.polygons, class_type+' not in polygons dictionary'
        assert class_type in self.polygons_predict, class_type+' not in polygons_predict dictionary'

        actual = self.polygons[class_type].multipolygon
        predicted   = self.polygons_predict[class_type].multipolygon
        tp = actual.intersection(predicted).area
        fp = predicted.area - tp
        fn = actual.area - tp
        return tp / (tp + fp + fn)

#Load a list of testing plots
def load_plots(plot_list, plot_type, image_types=config.image_types_to_load):
    all_plot_data=[]
    for i, plot_id in enumerate(plot_list):
        plot_data = plot_wrapper(plot_id)
        plot_data.plot_type=plot_type

        for image_type in image_types:
            if image_type=='hs':
                image_path = config.hs_image_dir + plot_id+'_hyper.tif'
            elif image_type=='chm':
                image_path = config.chm_image_dir + plot_id+'_chm.tif'

            plot_data.load_image(image_type, image_path)

        plot_data.set_transform('chm')
        # Load canopy shapefiles for training data
        if plot_type=='train':
            shapefile_path = config.training_polygons_dir+'ITC_'+plot_id+'.shp'
            plot_data.load_train_polygons(class_type = 'canopy',
                                          filename = shapefile_path)
        
        all_plot_data.append(plot_data)

    return all_plot_data
