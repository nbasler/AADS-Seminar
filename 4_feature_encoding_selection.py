# %%
# Import necessary libraries
import os
import gc
import re
import random as rd
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import importlib
import file_extraction
from image_preprocessing import crop_images_bbox, find_common_bbox
from feature_engineering import feature_extractor

# $$
# Config class
class Config:
    RANDOM_STATE = 42
    test_size = 0.1
    
config = Config()
# %%
# Prepare Data for Classifier and Train/Test Split______________________________________________________________

def prepare_data_classifier(dorl = 'D', # Args for extract_cmmd_dicom
                            cmmd_path=r'C:/Users/test/OneDrive/1_Uni/00_Master_Goethe/04_Semester/1_Advanced Applied Data Science (AADS)/AADS Code/Dataset/CMMD',
                            excel_path=r'C:/Users/test/OneDrive/1_Uni/00_Master_Goethe/04_Semester/1_Advanced Applied Data Science (AADS)/CMMD_clinicaldata_revision.xlsx',
                            resize_factor=0.25,
                            limit_dicom=0,
                            limit_feature=0, # Args for feature_extractor
                            workers=10,
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
                            morph_close_iter=5,
                            max_contours_per_type=200,  # CRITICAL: Limit contours processed
                            simplified_glcm=True,     # CRITICAL: Use single angle/distance
                            enable_texture_features=True,
                            skew_check=True,
                            ):
    """
    Function to prepare data for classification by extracting features from DICOM images,
    cropping images to a common bounding box, and splitting the data into training and test sets.
    For Arguments, see the docstring of the corresponding function
    
    Returns:
        - df_cmmd (pd.DataFrame): DataFrame containing the DICOM metadata with pixel arrray.
        - data_split (tuple): Tuple containing the training and test DataFrames.
        - features_df (pd.DataFrame): DataFrame containing the extracted features.
        - data_split (dict): Dictionary containing the training and test sets:
        - labels_df (pd.DataFrame): DataFrame containing the clinical data from the Excel file.
        - lassification_labels (pd.Series): Series containing the classification labels.
    """
    importlib.reload(file_extraction)
    
    # Load and extract DICOM files and clinical data from the CMMD dataset
    df_cmmd, labels_df = file_extraction.extract_cmmd_dicom(dorl = dorl,
                                            cmmd_path = cmmd_path,
                                            excel_path = excel_path,
                                            resize_factor = resize_factor,
                                            limit = limit_dicom)
    
    # crop images to commmon bounding box and store largest individual ROI
    img_dim_crop = {}
    df_cmmd['pixel_array'], img_dim_crop['width'], img_dim_crop['height'] = crop_images_bbox(df_cmmd['pixel_array'].tolist(),
                                                                                                            df_cmmd['image_laterality'].tolist())
    # Store largest ROI for each img in separate column of the dataframe
    df_cmmd['indv_bbox'] = find_common_bbox(df_cmmd['pixel_array'],df_cmmd['image_laterality'])
    
    # Limit the number of features to be extracted if limit_feature is set
    df_cmmd_subset = df_cmmd[:limit_feature] if limit_feature!=0 else df_cmmd
    
    # Extract features from the DICOM images using the feature_extractor function
    features_df = feature_extractor(df_cmmd_subset,
                                    workers = workers,
                                    hist_equalize = hist_equalize,
                                    chunksize = chunksize,
                                    df_image_column = df_image_column,
                                    df_id_column = df_id_column,
                                    quantize = quantize,
                                    laplace_fct = laplace_fct,
                                    sigma_gauss_calci = sigma_gauss_calci,
                                    sigma_gauss_mass = sigma_gauss_mass,
                                    gamma_calci = gamma_calci,
                                    gamma_mass = gamma_mass,
                                    kernel_mass = kernel_mass,
                                    max_thresh_contour_calci = max_thresh_contour_calci,
                                    max_thresh_contour_mass = max_thresh_contour_mass,
                                    morph_open_iter = morph_open_iter,
                                    morph_close_iter = morph_close_iter,
                                    max_contours_per_type = max_contours_per_type,  # CRITICAL: Limit contours processed
                                    simplified_glcm = simplified_glcm,     # CRITICAL: Use single angle/distance
                                    enable_texture_features = enable_texture_features  # Set False for 50x speedup
                                    )
    
    # Add features from df_cmmd_subset to features_df
    features_df.insert(0, 'code_meaning', df_cmmd_subset['code_meaning'])
    features_df.insert(0, 'laterality', df_cmmd_subset['image_laterality'])
    features_df.insert(0, 'age', df_cmmd_subset['Age']) 
    
    # Convert categorical columns to 'category' dtype first
    categorical_cols = features_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['code_meaning', 'laterality']:
            features_df[col] = features_df[col].astype('category')
    # Convert specific columns to category dtype
    if 'laterality' in features_df.columns:
        features_df['laterality'] = features_df['laterality'].astype('category')
    if 'code_meaning' in features_df.columns:
        features_df['code_meaning'] = features_df['code_meaning'].astype('category')

    # Encode categorical columns in features_df using OneHotEncoder
    categorical_cols = features_df.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        # Separate numerical and categorical data
        features_numerical = features_df.select_dtypes(exclude=['category', 'object'])
        features_categorical = features_df[categorical_cols]
        
        # Apply OneHotEncoder to categorical columns (no drop_first for subtype in features_df)
        for col in categorical_cols:
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(features_categorical[[col]])
            encoded_feature_names = encoder.get_feature_names_out([col])
            
            # Use format 'old-col name: category name' for column names
            category_names = [f"{col}: {name.replace(f'{col}_', '')}" for name in encoded_feature_names]
            encoded_df = pd.DataFrame(encoded_data, columns=category_names, index=features_df.index)
            
            # Add encoded columns to features_df
            features_df = pd.concat([features_df, encoded_df], axis=1)
        
        # Remove original categorical columns
        features_df = features_df.drop(columns=categorical_cols)
    
    print('\n', features_df.head())
    
    # Now create X from features_df for model training
    X = features_df.copy()

    # Identify numerical columns (before any categorical encoding) to check for skewness
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    # Check columns for skewness and apply log transformation if needed (only for original numericals)
    if skew_check and len(numeric_cols) > 0:
        # Exclude columns with 'kurtosis' or 'skewness' in their names
        skewed_cols = [col for col in numeric_cols if np.abs(X[col].skew()) > 0.5 and 'kurtosis' not in col.lower() and 'skewness' not in col.lower()]
        for col in skewed_cols:
            X[col] = np.log1p(X[col].clip(lower=-0.999))
            X[col] = X[col].astype('float32').round(4)
            
    # For X, apply drop_first to subtype columns to avoid dummy variable trap
    subtype_cols = [col for col in X.columns if col in df_cmmd_subset['subtype'].unique()]
    if len(subtype_cols) > 1:
        # Drop the first subtype column for X
        X = X.drop(columns=[subtype_cols[0]])
    
    # Convert DataFrame to numpy array for model training and ensure X
    # is a 2D array for model training input
    X = X.to_numpy()
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
            
    # Class, i.e. target variable y
    # classification_labels = df_cmmd['classification'].astype('category')
    # cat_encoder = OneHotEncoder(drop='first')  # Use drop='first' to avoid dummy variable trap
    # y = cat_encoder.fit_transform(df_cmmd_subset[['classification']]).toarray()
    # y = np.asarray(y).squeeze()
    # if y.ndim > 1:
    #     y = y.ravel()
    
    n_classes = df_cmmd_subset['classification'].nunique()
    if n_classes < 2:
        raise ValueError("Need at least two classes in 'classification' column for classification. Found only one.")

    cat_encoder = OneHotEncoder(drop='first')  # Use drop='first' to avoid dummy variable trap
    y = cat_encoder.fit_transform(df_cmmd_subset[['classification']]).toarray()
    y = np.asarray(y).squeeze()
    if y.ndim > 1:
        y = y.ravel()
    # Ensure y is not float but integer type
    if y.dtype == 'float64':
        y = y.astype('int32')

    # Store labels of class in case needed
    classification_labels = df_cmmd_subset['classification'].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size = config.RANDOM_STATE,
                                                        random_state = config.RANDOM_STATE,
                                                        shuffle = True,
                                                        stratify = y
                                                        )
    
    # Store train test split in a dictionary
    data_split = {'X_train': X_train,
                  'X_test': X_test,
                  'y_train': y_train,
                  'y_test': y_test
                  }
    
    return df_cmmd_subset, features_df, data_split, labels_df, classification_labels

# %%
# Feature Encoding_______________________________________________________________________________

# Obsolete function:
# Skewness check is now done in prepare_data_classifier function
# Standardization is done directly in the pipeline of the model training.
# def standardize_skew(feature_df, skew_check=True):
#     """
#     Applies log-transformation to skewed numerical features if skew_check is True.
#     And ensures data is standardiz
#     Returns:
#         - DataFrame with encoded and standardized features.
#     """
#     # Identify numerical columns (before any categorical encoding)
#     numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
#     # Check columns for skewness and apply log transformation if needed (only for original numericals)
#     if skew_check and len(numeric_cols) > 0:
#         # Exclude columns with 'kurtosis' or 'skewness' in their names
#         skewed_cols = [col for col in numeric_cols if np.abs(feature_df[col].skew()) > 0.5 and 'kurtosis' not in col.lower() and 'skewness' not in col.lower()]
#         for col in skewed_cols:
#             feature_df[col] = np.log1p(feature_df[col].clip(lower=-0.999))
#             feature_df[col] = feature_df[col].astype('float32').round(4)
            
#     # Now encode categorical columns in-place
#     categorical_cols = feature_df.select_dtypes(include=['category', 'object']).columns
#     if len(categorical_cols) > 0:
#         feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)
        
#     # Optimize float columns
#     float_cols = feature_df.select_dtypes(include='float').columns
#     feature_df[float_cols] = feature_df[float_cols].astype('float32').round(4)
#     feature_df = feature_df.fillna(feature_df.mean())
#     return feature_df

# %%
# Feature Selection_______________________________________________________________________________

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

def select_correlated_features(corr_matrix, target, threshold=0.1):
    """
    Select features highly correlated with the target variable.

    Parameters:
        corr_matrix (pd.DataFrame): Correlation matrix including the target.
        target (str): Name of the target column.
        threshold (float): Absolute correlation threshold for selection.

    Returns:
        selected_features (list): List of selected feature names.
        feature_select (pd.Series): Series of selected features and their correlations.
    """
    corr_with_target = corr_matrix[target].sort_values(ascending=False)
    corr_with_target = corr_with_target.drop(target)
    feature_select = corr_with_target[(corr_with_target > threshold) | (corr_with_target < -threshold)]
    selected_features = feature_select.index.tolist()
    return selected_features, feature_select

# Obsolete function since usual collinearity is an uissue for linear regresssion models,
# but for ML classification models this should not be a big issue.
# Hence we do not use VIF for feature selection, as it also just returns almost all features.
# def calculate_vif(X):
#     """
#     Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame X.
    
#     Parameters:
#     - X: DataFrame containing the features.
    
#     Returns:
#     - A DataFrame with VIF values for each feature.
#     """
#     vif_data = pd.DataFrame()
#     vif_data['feature'] = X.columns
#     vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     print(len(vif_data['VIF']), "features selected based on mutual information.")
#     return vif_data

def select_features_rfe(X, y, n_features=10, estimator=None):
    """
    Apply Recursive Feature Elimination (RFE) to select features.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): Target variable.
        n_features (int): Number of features to select.
        estimator: Estimator to use (default: LogisticRegression).

    Returns:
        - selected_features (pd.Index): Index of selected feature names.
    """
    if estimator is None:
        estimator = LogisticRegression(max_iter=10000)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    print(len(selected_features), "features selected based on mutual information.")
    return selected_features

def select_features_by_mutual_info(X, y, mi_threshold=0.1, discrete_features=True):
    """
    Select features based on mutual information.

    Parameters:
        - X (pd.DataFrame): Feature dataframe.
        - y (array-like): Target variable.
        - mi_threshold (float): Threshold for mutual information.
        - discrete_features (bool): Whether to treat features as discrete.
        
    Returns:
        selected_features (DataFrame): DF of selected feature names with MI values.
    """
    mi = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_df = pd.DataFrame({'feature': X.columns, 'mutual_info': mi})
    mi_df = mi_df.sort_values(by='mutual_info', ascending=False)
    selected_features = mi_df[mi_df['mutual_info'] > mi_threshold]
    print(len(selected_features), "features selected based on mutual information.")
    return selected_features