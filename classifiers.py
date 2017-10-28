import numpy as np
import pandas as pd


class watershed_classifier:
    def __init__(self):
        self.fitted_params={}
    
    def apply_classifier(self, plot, maxima_min_distance, ndvi_threshold,
                         max_crown_radius):
        # this is where I get nd
        labels, coordinates = self._labels_from_watershed(height_image = plot.images['height'].image_data,
                                                          ndvi_image = plot.images['ndvi'].image_data,
                                                          min_distance = maxima_min_distance,
                                                          ndvi_threshold = ndvi_threshold,
                                                          return_coordinates = True)
        buffer_circles = self._get_circles_from_points(array_shape=labels.shape, 
                                                       points=coordinates, 
                                                       radius=max_crown_radius)
        labels[~buffer_circles] = 0
        
        return(labels)

        
    def get_error(self, plots, **parameters):
        # gets the jaccard error given the parameters
        predicted_canopies=[]
        for p in plots:
            canopies = self.apply_classifier(p, **parameters)
            p.load_prediction(class_type='canopy',
                              predict = canopies,
                              reshape=False)
            
            
        
        # Make all canopies the same ID for the time being
        labels[labels!=0]=1
    
    def fit(self, plots):
        #Fits the parameters for doing watershed classification
        
        
        pass
    
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
        coordinates = peak_local_max(height_image, 
                                     min_distance=min_distance,
                                     labels=height_mask,
                                     threshold_abs=2, indices= True)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-chm_image, markers, mask=height_mask)
        
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