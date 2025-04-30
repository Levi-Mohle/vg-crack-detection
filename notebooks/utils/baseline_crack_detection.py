import numpy as np
import cv2
import pandas as pd
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import measure
from skimage.morphology import skeletonize, binary_dilation, binary_erosion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

def preprocess(height):
    height[height == 0] = height[height != 0].min()
    return height

def correct_positions(cracks_metadata, delta_x_px, delta_y_px):
    # TODO: implement position correction

    return cracks_metadata

def bwareaopen(binary_image, min_area):
    binary_image = np.asarray(binary_image, dtype=np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the thresholded image    
    mask = np.zeros_like(binary_image)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area: # Create a mask to keep contours with area greater than min_area
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask

def frangi_2d(image, sigma, beta):
        hess = hessian_matrix(image, sigma=sigma, use_gaussian_derivatives=True)
        lambda1, lambda2 = hessian_matrix_eigvals(hess)

        # Compute vesselness response
        Rb = lambda1 / (lambda2 + 1e-10)  # Adding a small constant to avoid division by zero
        S2 = np.sqrt(lambda1**2 + lambda2**2) # sum of squared eigenvalues of Hessian matrix

        return np.exp(-Rb**2 / (2 * beta**2)) * (1 - np.exp(-S2 / (2 * sigma**2))), Rb, S2

def frangi_filter(image, scale_range=(1, 10), scale_step=2, beta=0.5):

    result = np.zeros_like(image)
    Rb_total = np.zeros_like(image, dtype=np.float32)
    S2_total = np.zeros_like(image, dtype=np.float32)
    
    # Iterate over scales
    for scale in range(scale_range[0], scale_range[1] + 1, scale_step):
        # Apply Frangi filter at each scale
        frangi_response, Rb, S2 = frangi_2d(image, sigma=scale, beta=beta)
        result += frangi_response
        Rb_total += Rb
        S2_total += S2

    # Normalize result
    result = result / scale_range[1]
    Rb_total = Rb_total / scale_range[1]
    S2_total = S2_total / scale_range[1]

    return result, Rb_total, S2_total

def filter_eccentricity(image):
    regions = measure.regionprops(measure.label(image))
    filtered_mask = np.zeros_like(image, dtype=np.uint8)
    for region in regions:
        if region.eccentricity > 0.95:
            filtered_mask[region.coords[:,0], region.coords[:,1]] = 1
    return filtered_mask

def morphology_only(data_gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
    tophat_img = cv2.morphologyEx(data_gray,  cv2.MORPH_BLACKHAT, kernel) # perform blackhat operation
    
    height, width = data_gray.shape
    window_size = 400
    tophat_segmentation = np.zeros_like(data_gray, dtype=np.uint8)

    for i in range(0, height, window_size):
        for j in range(0, width, window_size):
            window = tophat_img[i:i+window_size, j:j+window_size]
            local_thresh = window.max() / 10
            local_segmentation = (window >= local_thresh).astype(np.uint8)
            tophat_segmentation[i:i+window_size, j:j+window_size] = local_segmentation
    
    BW = cv2.morphologyEx(tophat_segmentation.astype(np.uint8), cv2.MORPH_CLOSE, kernel) # closing operation 

    erod_dilate_image = binary_dilation(binary_erosion(BW)) # extra erosion/dilation operation for denoising
    result_image = bwareaopen(erod_dilate_image, min_area=20)
    filtered_mask = filter_eccentricity(result_image)

    return filtered_mask, tophat_img

def crack_detection(data_color, data_height):
    data_gray = cv2.cvtColor(data_color, cv2.COLOR_BGR2YCR_CB)[:,:,0] #get brightness
    mask_filter_gray_1, tophat_img = morphology_only(data_gray)
    filtered_image, Rb_total, S2_total = frangi_filter(data_height * 0.25, scale_range=(3, 10))
    frangi_segmentation = (filtered_image >= 0.01)

    mask_filter_frangi = np.zeros_like(data_gray)
    mask_filter_frangi[binary_dilation(frangi_segmentation)] = binary_dilation(mask_filter_gray_1)[binary_dilation(frangi_segmentation)] #combine color and height data
    filter_shape = filter_eccentricity(mask_filter_frangi)

    detected_mask = binary_dilation(binary_erosion(binary_dilation(filter_shape)))
    return detected_mask, Rb_total, S2_total, tophat_img

def analyze_components(crack_binary, Rb_total, S2_total, tophat_img):
    labels = measure.label(crack_binary, connectivity=2) # Perform connected component labeling
    regions = measure.regionprops(labels) # Extract region properties
    component_data = [{
        'Component': region.label - 1,
        'Mean_Rb': np.mean(Rb_total[labels == region.label]),
        'Mean_S2': np.mean(S2_total[labels == region.label]),
        'Eccentricity': region.eccentricity,
        'Mean_Tophat': np.mean(tophat_img[labels == region.label])
    } for region in regions]
    return pd.DataFrame(component_data), labels

def skeletonize_and_convert_to_polylines(crack_binary):
    skeleton_labels = measure.label(skeletonize(crack_binary), connectivity=2) # Perform connected component labeling on the skeletonized image
    return skeleton_labels, [region.coords for region in measure.regionprops(skeleton_labels)] # Extract polylines from the labeled skeletons
 
def nearest_neighbors(data, polylines):
    if not data.empty:
        scaler = StandardScaler() # Standardize the data
        scaled_data = scaler.fit_transform(data) 
        distances = euclidean_distances(scaled_data) # Compute the pairwise Euclidean distances
        data['most_similar'] = [list(np.argsort(row)[1:6]) for row in distances] # Find the 5 nearest neighbors for each component
        data['polylines_coordinates'] = polylines
        data['polylines_mercator'] = [[coord / 3072 for coord in polyline] for polyline in polylines]

    return data

def crack_detection_total(rgb, height):
    # height = preprocess(height)
    crack_detection_output = crack_detection(rgb, height) #crack detection
    component_data_tot, labels_tot = analyze_components(crack_detection_output[0], crack_detection_output[1], crack_detection_output[2], crack_detection_output[3]) #connected component analysis
    skeleton_labels_tot, polylines_tot = skeletonize_and_convert_to_polylines(crack_detection_output[0]) #Perform skeletonization and convert to polylines
    data_nearest = nearest_neighbors(component_data_tot, polylines_tot)

    return crack_detection_output[0], data_nearest
