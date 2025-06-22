# Data Processing for Mammography Image Analysis
# 
# This script contains all functions for data processing and preparation.
# It is imported and used by the model files.

import os
import gc
import re
import random as rd
import numpy as np
import pandas as pd
import pydicom as dcm
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
class Config:
    # Paths
    CWD_PATH = os.getcwd().replace('\\', '/')
    CMMD_PATH = r'C:\Users\Uni\Documents\01_Uni\02_SoSe 2025\Advanced Applied Data Science\manifest-1616439774456\CMMD'
    EXCEL_PATH = r'C:/Users/Uni/Documents/01_Uni/02_SoSe 2025/Advanced Applied Data Science/Cursor/CMMD_clinicaldata_revision.xlsx'
    
    # Data processing
    RESIZE_FACTOR = 0.25
    TARGET_SIZE = 128
    BATCH_SIZE = 50
    MAX_WORKERS = 6
    
    # Model
    CHANNELS = 1
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.2
    
    # Memory optimization
    OPTIMIZED_DTYPES = {
        'ID1': 'string[pyarrow]',
        'image_laterality': 'string[pyarrow]',
        'patient_orientation': 'string[pyarrow]',
        'code_meaning': 'string[pyarrow]',
        'Age': 'Int32',
        'number': 'Int32',
        'abnormality': 'string[pyarrow]',
        'classification': 'category',
        'subtype': 'category'
    }

config = Config()

def extract_dcm_metadata(dcm_path, img_ID1):
    """Extract metadata and image data from DICOM files."""
    try:
        dcm_obj = dcm.dcmread(dcm_path)
        patient_orientation = str(dcm_obj[0x0020, 0x0020])
        image_laterality = str(dcm_obj[0x0020, 0x0062])
        code_meaning = str(dcm_obj[0x0054, 0x0220][0][0x0008, 0x0104])
        pixel_array = cv2.resize(dcm_obj.pixel_array, None, fx=config.RESIZE_FACTOR, fy=config.RESIZE_FACTOR, interpolation=cv2.INTER_AREA)
        return [img_ID1, pixel_array, image_laterality, patient_orientation, code_meaning]
    except Exception as e:
        print(f"Failed to read DICOM file: {e}")
        return None

def collect_dicom_paths(base_path, limit=None):
    """Collect DICOM paths."""
    collected = []
    count = 0
    for subdirectory in os.listdir(base_path):
        sub_path = os.path.join(base_path, subdirectory)
        for dirpath, _, filenames in os.walk(sub_path):
            for filename in filenames:
                if filename.endswith(".dcm"):
                    collected.append((
                        os.path.join(dirpath, filename),
                        subdirectory))
                    count += 1
                    if limit and count >= limit:
                        print(f"Limit of {limit} DICOM files reached.")
                        return collected
    return collected

def process_dicom_batch(batch_files):
    """Process a batch of DICOM files in parallel."""
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = [
            executor.submit(extract_dcm_metadata, path, ID1) 
            for path, ID1 in batch_files
        ]
        return [f.result() for f in futures if f.result() is not None]

def preprocess_image_for_contours(image_array, norm_method='none'):
    """Optimized image preprocessing for contour detection."""
    if image_array is None:
        return None

    # Efficient normalization
    if norm_method == 'minmax':
        processed_image = cv2.normalize(
            image_array.astype(np.float32), 
            None, 0, 255, 
            cv2.NORM_MINMAX
        ).astype(np.uint8)
    elif norm_method == 'standard':
        processed_image = StandardScaler().fit_transform(
            image_array.reshape(-1, 1)
        ).reshape(image_array.shape)
        processed_image = cv2.normalize(
            processed_image.astype(np.float32), 
            None, 0, 255, 
            cv2.NORM_MINMAX
        ).astype(np.uint8)
    else:  # 'none'
        processed_image = (
            (image_array * 255).astype(np.uint8) 
            if image_array.max() <= 1.0 
            else image_array.astype(np.uint8)
        )
    
    # Ensure image is grayscale
    if processed_image.ndim == 3:
        processed_image = (
            cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            if processed_image.shape[2] == 3
            else processed_image[:,:,0]
        )
    
    return processed_image

def detect_largest_aoi_bbox_cv2(image_array, norm_method='none'):
    """Optimized detection of the largest area of interest."""
    if image_array is None:
        return None

    processed_image = preprocess_image_for_contours(image_array, norm_method)
    if processed_image is None or processed_image.size == 0:
        return None

    # Efficient contour detection
    _, thresh_image = cv2.threshold(
        processed_image, 0, 1, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh_image, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)

def find_aoi_bbox_cv2(image_list, lateralities_list, norm_method='none'):
    """Optimized detection of common areas of interest."""
    bboxes_params = []

    for img, lat in zip(image_list, lateralities_list):
        if img is None or img.size == 0:
            continue
        
        # Ensure image is 2D for shape attribute
        current_img_shape = img.shape
        if len(current_img_shape) < 2:
            print(f"Warning: Skipping image with unexpected shape {current_img_shape} in find_common_aoi_bbox_cv2.")
            continue
        img_height, img_width = current_img_shape[:2]
        
        bbox_local = detect_largest_aoi_bbox_cv2(img, norm_method)
        if bbox_local:
            x, y, w, h = bbox_local
            bboxes_params.append((x, y, w, h))

    if not bboxes_params:
        return None
    # Return the list of individual bounding boxes parameters to be stored in the DataFrame
    return bboxes_params

def crop_images_to_common_aoi_cv2(image_list, lateralities_list, norm_method='none'):
    """Optimized cropping of images to common areas of interest."""
    if not image_list:
        return []

    bboxes_params = find_aoi_bbox_cv2(image_list, lateralities_list, norm_method)
    if not bboxes_params:
        print("No valid bounding boxes found. Returning empty list.")
        return []
    
    # Extract widths from all bounding boxes and find maximum
    widths = [bbox[2] for bbox in bboxes_params]  # bbox[2] is width
    common_w = max(widths)
    
    common_x_left = 0
    common_x_right = image_list[0].shape[1]
    common_y = 0
    common_h = image_list[0].shape[0]
    print(f'Common bounding box dimensions using {norm_method}: w: {common_w}, h: {common_h}')
    
    cropped_images = []
    
    for img, lat in zip(image_list, lateralities_list):
        if img is None or img.size == 0:
            cropped_images.append(None)
            continue

        current_img_shape = img.shape
        if len(current_img_shape) < 2:
            print(f"Warning: Skipping image with unexpected shape {current_img_shape} in crop_images_to_common_aoi_cv2.")
            cropped_images.append(None)
            continue
        img_height, img_width = current_img_shape[:2]
        
        # Determine actual crop coordinates for the current image
        if lat == 'R':
            x_crop_start = common_x_right - common_w
            x_crop_end = common_x_right
        elif lat == 'L':
            x_crop_start = common_x_left
            x_crop_end = common_x_left + common_w

        # Perform the crop
        if img.ndim > 2:  # If it's color, crop all channels
            cropped_img = img[common_y:common_h, x_crop_start:x_crop_end, :]
        else:  # Grayscale
            cropped_img = img[common_y:common_h, x_crop_start:x_crop_end]

        cropped_images.append(cropped_img)
    
    return cropped_images

def load_and_preprocess_data():
    """Main function for data loading and preprocessing."""
    print("Loading and processing data...")
    
    # Collect DICOM paths (all 5202 files)
    dicom_paths = collect_dicom_paths(config.CMMD_PATH, limit=None)  # Process all files
    print(f"Found DICOM files: {len(dicom_paths)}")
    
    # Process files in batches
    all_data = []
    for i in tqdm(range(0, len(dicom_paths), config.BATCH_SIZE), desc="Processing batches"):
        batch = dicom_paths[i:i + config.BATCH_SIZE]
        batch_data = process_dicom_batch(batch)
        all_data.extend(batch_data)
        
        # Memory optimization: delete temporary data
        del batch_data
        gc.collect()
    
    # Convert to DataFrame
    df_arr = pd.DataFrame(all_data, columns=['ID1', 'pixel_array', 'image_laterality', 'patient_orientation', 'code_meaning'])
    
    # Optimize string extraction using vectorized functions
    def extract_laterality_code_meaning(x):
        return re.search(r"(?<= ')[^']+(?=')", x).group(0) \
            if pd.notnull(x) and re.search(r"(?<= ')[^']+(?=')", x) \
                else x

    def extract_orientation(x):
        return re.search(r"(?<=\[).*?(?=\])", x).group(0) \
            if pd.notnull(x) and re.search(r"(?<=\[).*?(?=\])", x) \
                else x

    # Apply in batch
    for col, func in zip(['image_laterality', 'patient_orientation', 'code_meaning'],
                         [extract_laterality_code_meaning, extract_orientation, extract_laterality_code_meaning]):
        df_arr[col] = df_arr[col].apply(func)
    
    # Load and merge Excel data
    df_excel = pd.read_excel(config.EXCEL_PATH, sheet_name='Sheet1')
    df_merged = pd.merge(df_arr, df_excel, on='ID1', how='left')
    df_merged.drop(columns=['LeftRight'], inplace=True, errors='ignore')
    
    # Downcast numerical columns and convert categorical/text to memory-efficient types
    opt_dtypes = {
        'ID1': 'string[pyarrow]',
        'image_laterality': 'string[pyarrow]',
        'patient_orientation': 'string[pyarrow]',
        'code_meaning': 'string[pyarrow]',
        'Age': 'Int32',
        'number': 'Int32',
        'abnormality': 'string[pyarrow]',
        'classification': 'category',
        'subtype': 'category'
    }

    for col, dtype in opt_dtypes.items():
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(dtype)
            if dtype == 'category':
                print(f"\nCategory mapping for column '{col}':")
                print(dict(enumerate(df_merged[col].cat.categories)))
    
    # Clean up memory
    del all_data, df_arr, df_excel
    gc.collect()
    
    return df_merged

def prepare_features_for_training(df):
    """Prepare features for training (only 'Age' as tabular feature)."""
    print("\nPreparing features for training...")
    
    # Extract and normalize image data
    X_pixels_flat = np.stack(
        df['pixel_array'].apply(lambda img: img.flatten())
    )
    
    # Calculate optimal image dimensions
    total_pixels = X_pixels_flat.shape[1]
    aspect_ratio = 1914 / 2294
    factors = [
        (i, total_pixels // i) 
        for i in range(1, int(np.sqrt(total_pixels)) + 1) 
        if total_pixels % i == 0
    ]
    sorted_factors = sorted(
        factors, 
        key=lambda x: abs(x[1]/x[0] - aspect_ratio)
    )
    ORIG_HEIGHT, ORIG_WIDTH = sorted_factors[0]
    
    # Calculate new dimensions
    IMG_HEIGHT = config.TARGET_SIZE
    IMG_WIDTH = int(config.TARGET_SIZE * (ORIG_WIDTH / ORIG_HEIGHT))
    
    # Reshape and resize images
    X_2d = X_pixels_flat.reshape(-1, ORIG_HEIGHT, ORIG_WIDTH)
    X_resized = np.array([
        cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) 
        for img in X_2d
    ])
    
    # Normalize images
    X_pixels_normalized = (
        X_resized.reshape(-1, IMG_HEIGHT, IMG_WIDTH, config.CHANNELS)
        .astype('float32') / 65535.0
    )
    
    # Extract and process tabular data (only 'Age')
    df_tabular = df[['Age']].copy()
    
    # Handle missing values in 'Age'
    if df_tabular['Age'].isnull().any():
        df_tabular['Age'] = df_tabular['Age'].fillna(df_tabular['Age'].mean())
        print("Missing values in 'Age' imputed with mean.")
    
    # Create preprocessing pipeline (numerical only)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age'])
        ]
    )
    
    # Process tabular data
    X_tabular_processed = preprocessor.fit_transform(df_tabular)
    
    # Process labels
    y = df['classification'].astype('category')
    y_encoded = pd.get_dummies(y, drop_first=True).values.ravel()
    
    return X_pixels_normalized, X_tabular_processed, y_encoded, IMG_HEIGHT, IMG_WIDTH

def get_train_test_split(X_pixels, X_tabular, y):
    """Perform train-test split and return split data."""
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    return {
        'X_train_pixels': X_pixels[train_idx],
        'X_test_pixels': X_pixels[test_idx],
        'X_train_tabular': X_tabular[train_idx],
        'X_test_tabular': X_tabular[test_idx],
        'y_train': y[train_idx],
        'y_test': y[test_idx]
    }

def process_data():
    """Main function that performs the entire data processing pipeline."""
    # Load and process raw data
    df_cmmd = load_and_preprocess_data()
    
    # Perform image preprocessing
    print("\nPerforming image preprocessing...")
    df_cmmd['indv_aoi_bbox_dim'] = find_aoi_bbox_cv2(
        df_cmmd['pixel_array'], 
        df_cmmd['image_laterality']
    )
    df_cmmd['pixel_array'] = crop_images_to_common_aoi_cv2(
        df_cmmd['pixel_array'].tolist(), 
        df_cmmd['image_laterality'].tolist(), 
        norm_method='minmax'
    )
    
    # Prepare features
    X_pixels, X_tabular, y, IMG_HEIGHT, IMG_WIDTH = prepare_features_for_training(df_cmmd)
    
    # Perform train-test split
    data_split = get_train_test_split(X_pixels, X_tabular, y)
    
    return df_cmmd, data_split, IMG_HEIGHT, IMG_WIDTH

if __name__ == "__main__":
    # Test data processing
    df_cmmd, data_split, IMG_HEIGHT, IMG_WIDTH = process_data()
    print("\nData processing completed:")
    print(f"DataFrame Shape: {df_cmmd.shape}")
    print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
    print("\nTraining data:")
    print(f"X_train_pixels: {data_split['X_train_pixels'].shape}")
    print(f"X_train_tabular: {data_split['X_train_tabular'].shape}")
    print(f"y_train: {data_split['y_train'].shape}") 