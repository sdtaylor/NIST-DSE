import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import watershed
from scipy import optimize

class watershed_classifier:
    def __init__(self):
        self.fitted_params={}
    
    def apply_classifier(self, plot, maxima_min_distance, ndvi_threshold,
                         max_crown_radius):
        #print(maxima_min_distance, ndvi_threshold, max_crown_radius)
        # this is where I get nd
        labels, coordinates = self._labels_from_watershed(height_image = plot.images['chm'].image_data,
                                                          ndvi_image = plot.images['ndvi'].image_data,
                                                          min_distance = maxima_min_distance,
                                                          ndvi_threshold = ndvi_threshold,
                                                          return_coordinates = True)
        buffer_circles = self._get_circles_from_points(array_shape=labels.shape, 
                                                       points=coordinates, 
                                                       radius=max_crown_radius)
        labels[~buffer_circles] = 0
        
        # Make all canopies the same ID for the time being
        labels[labels!=0]=1
        
        return(labels)

        
    def get_error(self, plots, **parameters):
        # gets the jaccard error given the parameters
        errors=[]
        for p in plots:
            canopies = self.apply_classifier(p, **parameters)
            p.load_prediction_mask(class_type='canopy',
                                   mask = canopies)
            errors.append(p.get_jaccard_error(class_type='canopy'))
        
        return(np.mean(errors))
    
    # A go between to translate the tuple from scipy.optimize functions
    # into a dictionary of model parameters
    def scipy_error(self,x):
        parameters = {'maxima_min_distance':int(x[0]),
                      'ndvi_threshold':x[1],
                      'max_crown_radius':int(x[2])}
        
        # Unreasonable parameters which throw cause an error get large error 
        # values
        try:
            error = self.get_error(plots=self.training_plots, **parameters) * -1
        except:
            error = 100
        
        return error
    
    def fit(self, plots):
        self.training_plots=plots
        #                   maxima_min_distance, ndvi_threshold, max_crown_radius
        parameter_ranges = (slice(1,20,1), slice(-1, 1,0.1), slice(1,40,1))
        
        optimized_results = optimize.brute(self.scipy_error, parameter_ranges)
        self.fitted_parameters = {'maxima_min_distance':int(optimized_results[0]),
                                  'ndvi_threshold':optimized_results[1],
                                  'max_crown_radius':int(optimized_results[2])}
    
        # Clear memory
        self.training_plots=None
        
    def predict(self, plots):
        #Applys the watershed classifer
        pass

    def _labels_from_watershed(self, height_image, ndvi_image, min_distance, ndvi_threshold,
                              return_coordinates=False):
        # Only do local maxima and watershedding where:
        height_mask = np.logical_or(ndvi_image>=ndvi_threshold, height_image>0)
        
        local_maxi = peak_local_max(height_image, 
                                     min_distance=min_distance,
                                     labels=height_mask,
                                     threshold_abs=2, indices= False)
        if return_coordinates:
            coordinates = peak_local_max(height_image, 
                                         min_distance=min_distance,
                                         labels=height_mask,
                                         threshold_abs=2, indices= True)
        
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-height_image, markers, mask=height_mask)
        
        if return_coordinates:
            return labels, coordinates
        else:
            return labels
        
    # Draw a circle in an array of size (array_x, array_y)
    # with circle center (center_x, center_y) and radius
    def _draw_circle(self,array_shape, circle_center ,radius):
        rows, cols = np.indices((array_shape[0], array_shape[1]))
        circle_center_row, circle_center_col = circle_center[0], circle_center[1]
        circle_mask = (rows - circle_center_row)**2 + (cols - circle_center_col)**2 < radius**2
        return circle_mask
    
    # Get a single image with masks for many (potentially overlapping)
    # circles. points are the center points for all circles
    def _get_circles_from_points(self, array_shape, points, radius):
        a = np.zeros(array_shape).astype(bool)
        for i in range(points.shape[0]):
            circle = self._draw_circle(array_shape, points[i], radius=radius)
            a = np.logical_or(a, circle)
        return a