#%%
# Loading necessary libraries for feature engineering

import os
import cv2
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import  ProcessPoolExecutor
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import laplace
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from functools import partial

#%%
# Config Class
class FeatureEngineeringConfig:
    def __init__(self, quantize=64, laplace_fct=0.2, sigma_gauss_calci=0.1, sigma_gauss_mass=0.3,
                 gamma_calci=0.3, gamma_mass=0.4, kernel_mass=2, max_thresh_contour_calci=3**2,
                 max_thresh_contour_mass=120**2, morph_open_iter=4, morph_close_iter=2,
                 max_contours_per_type=500, simplified_glcm=True, enable_texture_features=True):
        self.quantize = quantize
        self.laplace_fct = laplace_fct
        self.sigma_gauss_calci = sigma_gauss_calci
        self.sigma_gauss_mass = sigma_gauss_mass
        self.gamma_calci = gamma_calci
        self.gamma_mass = gamma_mass
        self.kernel_mass = kernel_mass
        self.max_thresh_contour_calci = max_thresh_contour_calci
        self.max_thresh_contour_mass = max_thresh_contour_mass
        self.morph_open_iter = morph_open_iter
        self.morph_close_iter = morph_close_iter
        self.max_contours_per_type = max_contours_per_type
        self.simplified_glcm = simplified_glcm
        self.enable_texture_features = enable_texture_features

#%%
# Feature engineering methods______________________________________________________________________________

def extract_contours(image_array,
                     laplace_fct=0.2,
                     sigma_gauss_calci=0.1,
                     sigma_gauss_mass=0.2,
                     gamma_calci=0.25,
                     gamma_mass=0.3,
                     kernel_mass=3,
                     max_thresh_contour_calci=3**2,
                     max_thresh_contour_mass=120**2,
                     morph_open_iter=3,
                     morph_close_iter=2,
                     max_contours_per_type=800
                     ):
    """
    Extracts contours of calcifications and masses from a breast cancer mammogram image.
    The contours are detected using OTSU thresholding and morphological operations in the image.
    Args:
        - image_array (2D numpy array): The gray-scale pixel data of the mammogram image.
        - laplace_fct (float): Factor for the Laplace filter to enhance edges.
        - sigma_gauss_calci: Standard deviation for Gaussian smoothing of calcification identification.
        - sigma_gauss_mass: Standard deviation for Gaussian smoothing of calcification identification.
                            Note: Mass identification profits from slightly higher sigma value than
                                  that of calcification.
        - gamma_calci: Gamma correction factor for calcification detection.
        - gamma_mass: Gamma correction factor for mass detection.
        - kernel_mass: Size of the kernel used for morphological operations on mass contours.
                    Note: Calcification Kernel is set to (1, 1) to preserve small structures.
        - max_thresh_contour_calci: Area threshold for calcification contours.
        - max_thresh_contour_mass: Area threshold for mass contours.
        - morph_open_iter: Number of iterations for morphological opening operation on mass contours.
        - morph_close_iter: Number of iterations for morphological closing operation on mass contours.
        - max_contours_per_type: Maximum number of contours to process per type (calcification/mass).
    Returns:
        - contours_calci: List of contours for calcifications.
        - contours_mass: List of contours for masses.
        - enhanced_img_calci: Enhanced image for calcification detection (if needed later).
        - enhanced_img_mass: Enhanced image for calcification detection (if needed later).
    """
    
    # Find calcification structures (Characterised by sharp edges, smaller and clustered regions;
    # depending on the cancer stage, calcification are of different size and forms).
    
    # Smoothing image using Gaussian filter to reduce noise and enhance edges by applying a Laplace filter.
    enhanced_img_calci = gaussian_filter(image_array, sigma=sigma_gauss_calci)
    enhanced_img_calci = adjust_gamma(enhanced_img_calci, gamma=gamma_calci)
    enhanced_img_calci = ((1-laplace_fct) * enhanced_img_calci  + laplace_fct * laplace(enhanced_img_calci))
    
    # Convert to uint8 for OpenCV threshold operation
    enhanced_img_calci = np.clip(enhanced_img_calci, 0, 255).astype(np.uint8)
    
    # Apply OTSU thresholding to find automatically thresholds to then create a binary image (balck and white),
    # which sets all pixels above the threshold to 255 (white) and below to 0 (black).
    _, thresh_img_calci = cv2.threshold(enhanced_img_calci, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise and fill holes we use a kernel of ones,
    # which is a 1x1 square that will be used to process the binary image.
    kernel_calci = np.ones((1, 1), np.uint8)
    thresh_img_calci = cv2.morphologyEx(thresh_img_calci,
                                        cv2.MORPH_OPEN,
                                        kernel_calci,
                                        iterations=morph_open_iter
                                        )
    thresh_img_calci = cv2.morphologyEx(thresh_img_calci,
                                        cv2.MORPH_CLOSE,
                                        kernel_calci,
                                        iterations=morph_close_iter
                                        )
    
    # Find calcification ROI contours including internal (RETR TREE instead of RETR EXTERNAL).
    # This ensures taht small contours nested within larger boxes, which might be neglected due to
    # set size constraint for calcification (i.e. max_thresh_contour_calci), are detected.
    contours_calci, _ = cv2.findContours(thresh_img_calci,
                                         cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE
                                         )
    
    # Filter conttours that are too large for calcification and limit the number of contours per image.
    contours_calci = [contour for contour in contours_calci if cv2.contourArea(contour) < max_thresh_contour_calci][:max_contours_per_type]
    
    # Find mass structures (Characterized flowing transient, wider regions and irregular shapes)
    # Smoothing image using Gaussian filter to reduce noise.
    enhanced_img_mass = gaussian_filter(image_array, sigma=sigma_gauss_mass)
    enhanced_img_mass = adjust_gamma(enhanced_img_mass, gamma = gamma_mass)
    
    # Convert to uint8 for OpenCV threshold operation
    enhanced_img_mass = np.clip(enhanced_img_mass, 0, 255).astype(np.uint8)
    
    # Apply OTSU thresholding to find automatically thresholds to then create a binary image (balck and white),
    # which sets all pixels above the threshold to 255 (white) and below to 0 (black).
    _, thresh_img_mass = cv2.threshold(enhanced_img_mass, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_mass = np.ones((3, 3), np.uint8)
    
    # Morphological operations to remove noise and fill holes we use a kernel of ones,
    # which is a small (but bigger than for calcification) square that will be used to process the binary image.
    # Further, for mass detection small objects, set to the size of the max threshold contour for calcification, are removed.
    thresh_img_mass = cv2.morphologyEx(thresh_img_mass,
                                       cv2.MORPH_OPEN,
                                       kernel_mass,
                                       iterations=morph_open_iter
                                       )
    thresh_img_mass = cv2.morphologyEx(thresh_img_mass,
                                       cv2.MORPH_CLOSE,
                                       kernel_mass,
                                       iterations=morph_close_iter
                                       )
    thresh_img_mass = remove_small_objects(thresh_img_mass,
                                           min_size=max_thresh_contour_calci
                                           )

    # Find contours including internal ones for mass ROIs
    contours_mass, _ = cv2.findContours(thresh_img_mass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_mass = [contour for contour in contours_mass if cv2.contourArea(contour) < max_thresh_contour_mass][:max_contours_per_type]

    return contours_calci, contours_mass, enhanced_img_calci, enhanced_img_mass, thresh_img_calci, thresh_img_mass
    
def extract_features_from_contours(image_array,
                                   quantize=64,
                                   laplace_fct=0.2,
                                   sigma_gauss_calci=0.1,
                                   sigma_gauss_mass=0.3,
                                   gamma_calci=0.3,
                                   gamma_mass=0.4,
                                   kernel_mass=2,
                                   max_thresh_contour_calci=3**2,
                                   max_thresh_contour_mass=120**2,
                                   morph_open_iter=3,
                                   morph_close_iter=2,
                                   max_contours_per_type=500,  # CRITICAL: Limit contours processed
                                   simplified_glcm=True,     # CRITICAL: Use single angle/distance
                                   enable_texture_features=True
                                   ):
    """
    Extracts features from contours of calcifications and masses in a breast cancer mammogram image.
    Contours are detecte using OTSU thresholding and morphological operations in the image.
    The function computes first-order, shape, and texture features that can be 
    then used as features for machine learning models.
    
    Parameters:
    - image_array: 2D numpy array representing the pixel data of the mammogram.
    - quantize: Number of gray levels the image is quantized to for GLCM (computational speedup if lower).
    - laplace_fct: Factor for the Laplace filter to enhance edges.
    - sigma_gauss_calci: Standard deviation for Gaussian smoothing of calcification identification.
    - sigma_gauss_mass: Standard deviation for Gaussian smoothing of calcification identification.
                        Note: Mass identification profits from slightly higher sigma value than
                              that of calcification.
    - gamma_calci: Gamma correction factor for calcification detection.
    - gamma_mass: Gamma correction factor for mass detection.
    - kernel_mass: Size of the kernel used for morphological operations on mass contours.
                   Note: Calcification Kernel is set to (1, 1) to preserve small structures.
    - max_thresh_contour_calci: Area threshold for calcification contours.
    - max_thresh_contour_mass: Area threshold for mass contours.
    - morph_open_iter: Number of iterations for morphological opening operation on mass contours.
    - morph_close_iter: Number of iterations for morphological closing operation on mass contours.
    - max_contours_per_type: Maximum number of contours to process per type (calcification/mass).
    - simplified_glcm: If True, uses a single angle and distance for GLCM (faster).
    - enable_texture_features: If True, computes texture features using GLCM.
    Returns:
    - features_df: DataFrame containing the extracted features for an image.
    """

    contours_calci, contours_mass, _, _, _, _ = extract_contours(image_array,
                                                                 sigma_gauss_calci=sigma_gauss_calci,
                                                                 sigma_gauss_mass=sigma_gauss_mass,
                                                                 gamma_calci=gamma_calci,
                                                                 gamma_mass=gamma_mass,
                                                                 laplace_fct=laplace_fct,
                                                                 kernel_mass=kernel_mass,
                                                                 morph_open_iter=morph_open_iter,
                                                                 morph_close_iter=morph_close_iter,
                                                                 max_thresh_contour_calci=max_thresh_contour_calci,
                                                                 max_thresh_contour_mass=max_thresh_contour_mass,
                                                                 max_contours_per_type=max_contours_per_type
                                                                )   
    
    # Initialize lists to store features
    features_list_calci = []
    features_list_mass = []
    
    # Configuration for GLCM (Gray Level Co-occurrence Matrix)
    # Quantize image for GLCM to reduce noise and speed up calculations (at cost of information loss).
    
    if quantize == 32:
        image_quant = (image_array / image_array.max() * 31).astype(np.uint8)
        set_level = 32
    elif quantize == 64:
        image_quant = (image_array / image_array.max() * 63).astype(np.uint8)
        set_level = 64
    else:
        image_quant = image_array.astype(np.uint8)
        set_level = 256  # Use full range for GLCM
    
    # Multi-angle and -length GLCM for rotation-invariant texture analysis
    if simplified_glcm:
        angles = [0]  # Single angle for 4x speedup
        distances = [1]  # Single distance for 2x speedup
    else:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Full angles
        distances = [1, 2]  # Multiple distances for better texture characterization
    
    # Start two separate loops for calcification and mass contours to extract features separately
    # due to the different nature of the structures (area, shape, contrast, texture)
    
    # Loop through calcification contours
    for i, contour in enumerate(contours_calci):
        # 1. Create binary mask from contour
        mask = np.zeros(image_array.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, color=1, thickness=-1)

        # 2. First-order features
        pixels = image_array[mask > 0]
        first_order_calci = {
            'roi_id': i,
            'count_px_calci': len(pixels),
            'mean_px_calci': np.mean(pixels),
            'std_px_calci': np.std(pixels),
            'min_px_calci': np.min(pixels),
            'max_px_calci': np.max(pixels),
            'skewness_px_calci': safe_moment_calc(skew, pixels),
            'kurtosis_px_calci': safe_moment_calc(kurtosis, pixels),
            # 'energy_px_calci': np.sum(pixels.astype(np.float64)**2),  # Use float64 for calculation
            # 'entropy_px_calci': safe_entropy_calc(pixels)
        }
        
        # 3. Shape features (from regionprops)
        region = label(mask)
        props = regionprops(region)[0]
        shape_calci = {
            'area_calci': props.area,
            'perimeter_calci': props.perimeter,
            'eccentricity_calci': props.eccentricity,
            'solidity_calci': props.solidity,
            'extent_calci': props.extent
        }

        # 4. Texture features (GLCM) - OPTIONAL for speed
        if enable_texture_features:
            image_masked = image_quant * mask
            glcm_calci = graycomatrix(image_masked, distances=distances, angles=angles, 
                               levels=set_level, symmetric=True, normed=True)
            # Calculate mean across all angles and distances for rotation-invariant features
            # Compute variance and entropy manually, since the arguments won't work (although in documentation)

            texture_calci = {
                'contrast_texture_calci': np.mean(graycoprops(glcm_calci, 'contrast')),
                'homogeneity_texture_calci': np.mean(graycoprops(glcm_calci, 'homogeneity')),
                'dissimilarity_texture_calci': np.mean(graycoprops(glcm_calci, 'dissimilarity')),
                'energy_texture_calci': np.mean(graycoprops(glcm_calci, 'energy')),
                'correlation_texture_calci': np.mean(graycoprops(glcm_calci, 'correlation')),
                'ASM_texture_calci': np.mean(graycoprops(glcm_calci, 'ASM')),
                'entropy_texture_calci': (-np.sum(glcm_calci * np.log2(glcm_calci + 1e-10))),
                'variance_texture_calci': np.var(glcm_calci)
            }
            
            # 5. Non-linear GLCM features
            # Calculate GLCM features for skewness and kurtosis
            glcm_calci_flat = glcm_calci.flatten()
            if np.any(glcm_calci_flat):
                texture_calci['glcm_calci_skewness'] = safe_moment_calc(skew, glcm_calci_flat[glcm_calci_flat > 0])
                texture_calci['glcm_calci_kurtosis'] = safe_moment_calc(kurtosis, glcm_calci_flat[glcm_calci_flat > 0])
            else:
                texture_calci['glcm_calci_skewness'] = 0.0
                texture_calci['glcm_calci_kurtosis'] = 0.0
            
            # Calculate GLCM ln(1+X) features
            texture_calci['log_contrast_texture_calci'] = np.log1p(texture_calci['contrast_texture_calci'])
            texture_calci['log_homogeneity_texture_calci'] = np.log1p(texture_calci['homogeneity_texture_calci'])
            texture_calci['log_dissimilarity_texture_calci'] = np.log1p(texture_calci['dissimilarity_texture_calci'])
            texture_calci['log_correlation_texture_calci'] = np.log1p(texture_calci['correlation_texture_calci'])
        else:
            # Skip texture features for ultra speed - set to 0
            texture_calci = {
                'contrast_texture_calci': 0.0, 'homogeneity_texture_calci': 0.0,
                'dissimilarity_texture_calci': 0.0, 'energy_texture_calci': 0.0,
                'correlation_texture_calci': 0.0, 'ASM_texture_calci': 0.0,
                'entropy_texture_calci': 0.0, 'variance_texture_calci': 0.0,
                'glcm_calci_skewness': 0.0, 'glcm_calci_kurtosis': 0.0,
                'log_contrast_texture_calci': 0.0, 'log_homogeneity_texture_calci': 0.0,
                'log_dissimilarity_texture_calci': 0.0, 'log_correlation_texture_calci': 0.0
            }

        # Combine all features
        roi_features_calci = {**first_order_calci, **shape_calci, **texture_calci}
        features_list_calci.append(roi_features_calci)
        
    if not features_list_calci:
        # No calcification contours found, create empty features with 0 values
        # This ensures that the DataFrame has the same structure even if no contours are found
        calci_feature_names = [
            'num_roi_calci', 'count_px_calci', 'mean_px_calci', 'std_px_calci', 'min_px_calci',
            'max_px_calci', 'skewness_px_calci', 'kurtosis_px_calci',
            'area_calci', 'perimeter_calci', 'eccentricity_calci', 'solidity_calci', 'extent_calci',
            'variance_texture_calci', 'contrast_texture_calci', 'homogeneity_texture_calci',
            'dissimilarity_texture_calci', 'energy_texture_calci', 'correlation_texture_calci',
            'entropy_texture_calci', 'glcm_calci_skewness', 'glcm_calci_kurtosis',
            'log_contrast_texture_calci', 'log_homogeneity_texture_calci', 'log_dissimilarity_texture_calci',
            'log_correlation_texture_calci'
        ]
        features_dict_calci = {name: 0.0 for name in calci_feature_names}
        features_df_calci = pd.DataFrame([features_dict_calci])
    else:
        # Convert list of features to DataFrame and calculate mean over all ROIs
        features_df_calci = pd.DataFrame(features_list_calci)
        features_dict_calci = features_df_calci.mean().to_dict()
        features_df_calci = pd.DataFrame([features_dict_calci])  # Convert back to DataFrame with a single row
        features_df_calci.pop('roi_id') #Drop ROI id since not needed in final features
        features_df_calci.insert(0, 'num_roi_calci', len(features_list_calci))  # Add number of ROIs as first column
    
    # Loop through mass contours
    for i, contour in enumerate(contours_mass):
        # 1. Create binary mask from contour
        mask = np.zeros(image_array.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, color=1, thickness=-1)
        
        # 2. First-order features
        pixels = image_array[mask > 0]
        first_order_mass = {
            'roi_id': i,
            'count_px_mass': len(pixels),
            'mean_px_mass': np.mean(pixels), 
            'std_px_mass': np.std(pixels),
            'min_px_mass': np.min(pixels),
            'max_px_mass': np.max(pixels),
            'skewness_px_mass': safe_moment_calc(skew, pixels),
            'kurtosis_px_mass': safe_moment_calc(kurtosis, pixels),
            # 'energy_px_mass': np.sum(pixels.astype(np.float64)**2),
            # 'entropy_px_mass': safe_entropy_calc(pixels)
        }
        
        # 3. Shape features (from regionprops)
        region = label(mask)
        props = regionprops(region)[0]
        shape_mass = {
            'area_mass': props.area,
            'perimeter_mass': props.perimeter,
            'eccentricity_mass': props.eccentricity,
            'solidity_mass': props.solidity,
            'extent_mass': props.extent
        }
        
        # 4. Texture features (GLCM) - OPTIONAL for speed
        if enable_texture_features:
            image_masked = image_quant * mask
            glcm_mass = graycomatrix(image_masked, distances=distances, angles=angles, 
                               levels=set_level, symmetric=True, normed=True)
            # Calculate mean across all angles and distances for rotation-invariant features
            # Compute variance and entropy manually, since the arguments won't work (although in documentation)

            texture_mass = {
                'contrast_texture_mass': np.mean(graycoprops(glcm_mass, 'contrast')),
                'homogeneity_texture_mass': np.mean(graycoprops(glcm_mass, 'homogeneity')),
                'dissimilarity_texture_mass': np.mean(graycoprops(glcm_mass, 'dissimilarity')),
                'energy_texture_mass': np.mean(graycoprops(glcm_mass, 'energy')),
                'correlation_texture_mass': np.mean(graycoprops(glcm_mass, 'correlation')),
                'ASM_texture_mass': np.mean(graycoprops(glcm_mass, 'ASM')),
                'entropy_texture_mass': (-np.sum(glcm_mass * np.log2(glcm_mass + 1e-10))),
                'variance_texture_mass': np.var(glcm_mass)
            }
            
            # 5. Non-linear GLCM features
            # Calculate GLCM features for skewness and kurtosis
            glcm_mass_flat = glcm_mass.flatten()
            if np.any(glcm_mass_flat):
                glcm_mass_skewness = safe_moment_calc(skew, glcm_mass_flat[glcm_mass_flat > 0])
                glcm_mass_kurtosis = safe_moment_calc(kurtosis, glcm_mass_flat[glcm_mass_flat > 0])
            else:
                glcm_mass_skewness = 0.0
                glcm_mass_kurtosis = 0.0
            texture_mass['glcm_mass_skewness'] = glcm_mass_skewness
            texture_mass['glcm_mass_kurtosis'] = glcm_mass_kurtosis
            
            # Calculate GLCM ln(1+X) features
            texture_mass['log_contrast_texture_mass'] = np.log1p(texture_mass['contrast_texture_mass'])
            texture_mass['log_homogeneity_texture_mass'] = np.log1p(texture_mass['homogeneity_texture_mass'])
            texture_mass['log_dissimilarity_texture_mass'] = np.log1p(texture_mass['dissimilarity_texture_mass'])
            texture_mass['log_correlation_texture_mass'] = np.log1p(texture_mass['correlation_texture_mass'])
        else:
            # Skip texture features for ultra speed - set to 0
            texture_mass = {
                'contrast_texture_mass': 0.0, 'homogeneity_texture_mass': 0.0,
                'dissimilarity_texture_mass': 0.0, 'energy_texture_mass': 0.0,
                'correlation_texture_mass': 0.0, 'ASM_texture_mass': 0.0,
                'entropy_texture_mass': 0.0, 'variance_texture_mass': 0.0,
                'glcm_mass_skewness': 0.0, 'glcm_mass_kurtosis': 0.0,
                'log_contrast_texture_mass': 0.0, 'log_homogeneity_texture_mass': 0.0,
                'log_dissimilarity_texture_mass': 0.0, 'log_correlation_texture_mass': 0.0
            }

        # Combine all features
        roi_features_mass = {**first_order_mass, **shape_mass, **texture_mass}
        features_list_mass.append(roi_features_mass)

        # Combine all features
        roi_features_mass = {**first_order_mass, **shape_mass, **texture_mass}
        features_list_mass.append(roi_features_mass)
    
    # Calculate the mean of each feature across all ROIs for masses
    if not features_list_mass:
        # No mass contours found, create empty features with default values
        mass_feature_names = [
            'num_roi_mass', 'count_px_mass', 'mean_px_mass', 'std_px_mass', 'min_px_mass', 'max_px_mass',
            'skewness_px_mass', 'kurtosis_px_mass', 'area_mass', 'perimeter_mass', 'eccentricity_mass',
            'solidity_mass', 'extent_mass', 'variance_texture_mass', 'contrast_texture_mass',
            'homogeneity_texture_mass', 'dissimilarity_texture_mass', 'energy_texture_mass', 'correlation_texture_mass',
            'ASM_texture_mass', 'entropy_texture_mass', 'glcm_mass_skewness', 'glcm_mass_kurtosis',
            'log_contrast_texture_mass', 'log_homogeneity_texture_mass', 'log_dissimilarity_texture_mass',
            'log_correlation_texture_mass'
        ]
        features_dict_mass = {name: 0.0 for name in mass_feature_names}
        features_df_mass = pd.DataFrame([features_dict_mass])
    else:
        features_df_mass = pd.DataFrame(features_list_mass)
        features_dict_mass = features_df_mass.mean().astype(np.float16).to_dict()
        features_df_mass = pd.DataFrame([features_dict_mass])  # Convert back to DataFrame with a single row
        features_df_mass.pop('roi_id')
        features_df_mass.insert(0, 'num_roi_mass', len(features_list_mass))  # Add number of ROIs as first column
    
    # Merge features from calcifications and masses
    features_df = pd.concat([features_df_calci, features_df_mass], axis=1)
    
    # Round all float columns to 4 decimals
    features_df = features_df.round(4)

    # (Optional) Convert all float columns to float32 to save memory
    float_cols = features_df.select_dtypes(include='float').columns
    features_df[float_cols] = features_df[float_cols].astype('float32')
    
    return features_df

def feature_extractor(target_df,
                      workers=None,
                      hist_equalize='adaptive',
                      chunksize=None,
                      df_image_column='pixel_array',
                      df_id_column='ID1',
                      quantize=64,
                      laplace_fct=0.2,
                      sigma_gauss_calci=0.1,
                      sigma_gauss_mass=0.2,
                      gamma_calci=0.25,
                      gamma_mass=0.3,
                      kernel_mass=3,
                      max_thresh_contour_calci=3**2,
                      max_thresh_contour_mass=120**2,
                      morph_open_iter=3,
                      morph_close_iter=2,
                      max_contours_per_type=500,  # CRITICAL: Limit contours processed
                      simplified_glcm=True,     # CRITICAL: Use single angle/distance
                      enable_texture_features=True  # Set False for 50x speedup
                      ):
    """
    Extracts features from the contours of all images in the given list in the DataFrame utilizinge
    the functions `extract_contours` and `extract_features_from_contours` above.
    
    Args:
    - target_df: DataFrame containing the images.
    - workers: Number of worker processes (None for auto-detection based on CPU cores)
    - df_image_column: Name of the column containing the pixel arrays.
    - chunksize: Number of images per chunk (None for auto-calculation)
    
    Returns:
    - A DataFrame with the features extracted from the contours.
    """
    
    if df_image_column not in target_df.columns:
        raise ValueError(f"Column '{df_image_column}' not found in DataFrame.")
    
    if hist_equalize == 'equal':
        # Apply histogram equalization to enhance contrast
        gamma_calci = 0.2
        gamma_mass = 0.3
        target_df.loc[:, df_image_column] = target_df.loc[:, df_image_column].apply(lambda x: cv2.equalizeHist(x.astype(np.uint8)))
    elif hist_equalize == 'adaptive':
        # Apply adaptive histogram equalization (CLAHE)
        # Although not the best choice for mammograms acc. to this paper (https://www.sciencedirect.com/science/article/pii/S1877050919321519),
        # but still better than no or equalization of histogram.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # apply CLAHE to each image in the list of images
        target_df.loc[:, df_image_column] = target_df.loc[:, df_image_column].apply(lambda x: clahe.apply(x.astype(np.uint8)))    
    
    # Optimize number of workers based on CPU cores with a minimum of 8 if possible,
    # else the actual number of cores minus one to leave one core free for the system.
    if workers is None:
        workers = min(mp.cpu_count(), 10)
        print(f"Auto-detected {workers} workers")
    
    # Auto-calculate optimal chunk size
    if chunksize is None:
        chunksize = max(1, len(target_df) // (workers * 2))
        print(f"Using chunksize: {chunksize}")
    
    # Create partial function with fixed parameters for cleaner multiprocessing
    extract_func = partial(
        extract_features_from_contours,
        quantize=quantize,
        laplace_fct=laplace_fct,
        sigma_gauss_calci=sigma_gauss_calci,
        sigma_gauss_mass=sigma_gauss_mass,
        gamma_calci=gamma_calci,
        gamma_mass=gamma_mass,
        kernel_mass=kernel_mass,
        max_thresh_contour_calci=max_thresh_contour_calci,
        max_thresh_contour_mass=max_thresh_contour_mass,
        morph_open_iter=morph_open_iter,
        morph_close_iter=morph_close_iter,
        max_contours_per_type=max_contours_per_type,
        simplified_glcm=simplified_glcm,
        enable_texture_features=enable_texture_features
    )

    # Use ProcessPoolExecutor with optimized settings for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Use map() instead of submit() for better performance with large datasets
        features_list = list(executor.map(extract_func,
                                          target_df[df_image_column],
                                          chunksize=chunksize
                                          ))
    
    # Filter out None results and concatenate
    valid_features = [f for f in features_list if f is not None]
    
    if not valid_features:
        print("No features extracted.")
        return pd.DataFrame()

    features_df = pd.concat(valid_features, axis=0, ignore_index=True)
    
    return features_df

# Helper functions to avoid warnings in loops______________________________________________________________________

def adjust_gamma(image_array, gamma=0.5):
    """
    Adjusts the gamma, responsible for the brightness/contrast of the image.
    Args:
        - image_array (numpy.ndarray): Input image array
        - gamma (float): Gamma factor to adjust the image brightness/contrast (is used as inverse!)
    Return:
        - numpy.ndarray: Gamma-adjusted image
    """
    inv_Gamma = 1.0 / gamma
    look_up_table = np.array([((i / 255.0) ** inv_Gamma) * 255 for i in range(256)], dtype=np.uint8)
    
    # Uses openCV's LUT (Look-Up Table) function to apply the gamma correction    
    return cv2.LUT(image_array.astype(np.uint8), look_up_table)

def safe_moment_calc(moment_func, data):
    """
    Safely calculate statistical moments (skewness, kurtosis) to avoid warnings
    when data values are nearly identical.
    """
    try:
        # Check if data has sufficient variation
        if len(np.unique(data)) <= 2 or np.std(data) < 1e-10:
            return 0.0  # Return neutral value for nearly identical data
        
        result = moment_func(data)
        # Check for invalid results
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return float(result)
    except:
        return 0.0

def safe_entropy_calc(pixels):
    """
    Safely calculate entropy to avoid overflow warnings.
    """
    try:
        # Ensure pixels are positive and normalized
        p = np.clip(p, 1e-10, 1)
        entropy = -np.sum(p * np.log2(p))
        
        # Check for overflow/invalid results
        if np.isnan(entropy) or np.isinf(entropy):
            return 0.0
        return float(entropy)
    except:
        return 0.0

def create_empty_features_dict():
    """Return empty features when no contours found - for performance"""
    features = {}
    
    # Calcification features
    features['num_roi_calci'] = 0
    calci_features = ['count_px_calci', 'mean_px_calci', 'std_px_calci', 'min_px_calci',
                      'max_px_calci', 'skewness_px_calci', 'kurtosis_px_calci', 'area_calci', 'perimeter_calci', 'eccentricity_calci', 
                      'solidity_calci', 'extent_calci', 'contrast_texture_calci', 'homogeneity_texture_calci',
                      'dissimilarity_texture_calci', 'energy_texture_calci', 'correlation_texture_calci',
                      'ASM_texture_calci', 'entropy_texture_calci', 'variance_texture_calci']
    
    # Mass features  
    features['num_roi_mass'] = 0
    mass_features = ['count_px_mass', 'mean_px_mass', 'std_px_mass', 'min_px_mass',
                     'max_px_mass', 'skewness_px_mass', 'kurtosis_px_mass', 'area_mass', 'perimeter_mass', 'eccentricity_mass',
                     'solidity_mass', 'extent_mass', 'contrast_texture_mass', 'homogeneity_texture_mass',
                     'dissimilarity_texture_mass', 'energy_texture_mass', 'correlation_texture_mass',
                     'ASM_texture_mass', 'entropy_texture_mass', 'variance_texture_mass']
    
    # Initialize all with 0
    for feature in calci_features + mass_features:
        features[feature] = 0.0
        
    return features