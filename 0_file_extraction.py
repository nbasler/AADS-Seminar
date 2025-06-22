import os
import re
import pandas as pd
import cv2
import pydicom as dcm
from concurrent.futures import ThreadPoolExecutor

# Extraction methods______________________________________________________________________________

def extract_cmmd_dicom(dorl = 'D',
                       cmmd_path=r'C:/Users/test/OneDrive/1_Uni/00_Master_Goethe/04_Semester/1_Advanced Applied Data Science (AADS)/AADS Code/Dataset/CMMD',
                       excel_path=r'C:/Users/test/OneDrive/1_Uni/00_Master_Goethe/04_Semester/1_Advanced Applied Data Science (AADS)/CMMD_clinicaldata_revision.xlsx',
                       resize_factor=0.25,
                       limit=0):
    """
    Extracts pixel data and metadata from DICOM files (medical file type; .dcm) in the CMMD dataset and 
    merges it with the accompanying clinical data excel. Metadata includes patient orientation, image laterality, and code meaning.
    The resulting DataFrame is returned, containing all the collected data per patient (ID1) per image (2x or 4x per ID1),
    yielding n(ID1)xn(images) rows.
    
    Args:
        - dorl (str): 'D' if loading on Desktop, 'L' if on Laptop (specific for Norman's machines)
        - resize_factor (float): Factor by which to resize the DICOM pixel arrays to save memory later on.
        - limit (str or int): Limit on the number of DICOM files to process. If 'none', all files are processed.
        - cmmd_path (str): Path to the CMMD DICOM folder.
        - excel_path (str): Is assumed to be at the current working directory and contains clinical data.
    Returns:
        - pd.DataFrame: A DataFrame containing the merged DICOM pixel data, metadata and the Excel clinical data.
    """

    # If on Norman's Desktop, change the paths to the CMMD and Excel files
    if cmmd_path == r'C:/Users/test/OneDrive/1_Uni/00_Master_Goethe/04_Semester/1_Advanced Applied Data Science (AADS)/AADS Code/Dataset/CMMD':
        if dorl == 'D':
            cmmd_path = cmmd_path.replace('test', 'norma')
            excel_path = excel_path.replace('test', 'norma')
    
    # Replace backslashes with forward slashes for compatibility
    cmmd_path = cmmd_path.replace('\\', '/')
    excel_path = excel_path.replace('\\', '/')
            
    def extract_dcm_metadata(dcm_path, img_ID1):
        """ Extracts metadata and pixel data from a DICOM file."""
        
        try:
            # Read the DICOM file
            dcm_obj = dcm.dcmread(dcm_path)
            # Retrieve Patient Orientation relative to the Mammography-Device
            patient_orientation = str(dcm_obj[0x0020, 0x0020])
            # Retrieve Image Laterality (Left/Right)
            image_laterality = str(dcm_obj[0x0020, 0x0062])
            # Retrieve Code Meaning ('MLO' for Multi-Lateral Oblique, 'CC' for Cranio-Caudal)
            code_meaning = str(dcm_obj[0x0054, 0x0220][0][0x0008, 0x0104])
            # Retrieve pixel array and resize it using openCV lib; INTER_ARE interpolation method is used for downscaling to the factor specified
            pixel_array = cv2.resize(dcm_obj.pixel_array, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            return [img_ID1, pixel_array, image_laterality, patient_orientation, code_meaning]
        except Exception as e:
            # raise exception if loading DICOM file fails
            print(f"Failed to read DICOM file: {e}")
            return None

    def collect_dicom_paths(base_path, limit=0):
        """ Collects paths to all DICOM files in the specified CMMD folder which consist of several subfolders for each ID1 and pictures."""
        
        collected = []
        
        for count, subdirectory  in enumerate(os.listdir(base_path)):
            # create subdirectory path
            sub_path = os.path.join(base_path, subdirectory)
            # get subdirectories and filenames, neglect directory names ('_')
            for dirpath, _, filenames in os.walk(sub_path):
                for filename in filenames:
                    # loop through all files in the subdirectories and collect their paths if DICOM files (.dcm)
                    if filename.endswith(".dcm"):
                        collected.append((
                            os.path.join(dirpath, filename),
                            subdirectory))
                        count += 1
                        if limit == 0:
                            continue
                        elif count >= limit:
                            return collected
        return collected

    # Parallel loading using ThreadPoolExecutor to speed up the extraction process using the before defined functions
    dicom_paths = collect_dicom_paths(cmmd_path, limit=limit)
    data = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(extract_dcm_metadata, path, ID1) for path, ID1 in dicom_paths]
        for future in futures:
            result = future.result()
            if result:
                data.append(result)

    # Convert list to DataFrame
    df_data = pd.DataFrame(data, columns=['ID1', 'pixel_array', 'image_laterality', 'patient_orientation', 'code_meaning'])

    # Optimize string extraction using regular expressions to only get the relevant parts of the strings,
    # as the DICOM metadate is stored between single quotes and/or in square brackets which we do not want to store.
    def extract_laterality_code_meaning(x):
        return re.search(r"(?<= ')[^']+(?=')", x).group(0) \
            if pd.notnull(x) and re.search(r"(?<= ')[^']+(?=')", x) \
                else x

    def extract_orientation(x):
        return re.search(r"(?<=\[).*?(?=\])", x).group(0) \
            if pd.notnull(x) and re.search(r"(?<=\[).*?(?=\])", x) \
                else x

    # Extract/clean the string information per column in one batch per column
    for col, func in zip(['image_laterality', 'patient_orientation', 'code_meaning'],
                         [extract_laterality_code_meaning, extract_orientation, extract_laterality_code_meaning]):
                         df_data[col] = df_data[col].apply(func)

    # Merge DataFrame with all relevant data with Excel data on the ID1 column which is in both DataFrames
    df_excel = pd.read_excel(excel_path, sheet_name='Sheet1')
    df_merged = pd.merge(df_data, df_excel, on='ID1', how='left')
    df_merged.drop(columns=['LeftRight'], inplace=True, errors='ignore') # Not needed as loaded from dicom metadata

    # Downcast numerical columns and convert categorical/text to memory-efficient types.
    # To reduced memory overhead, we convert some 'object' dtype columns only containing strings into 'string[pyarrow]' dtypes.
    # This is more memory efficient than the default 'object' dtype used by pandas to store strings in DFs.
    
    # fill missing values in subtype with 'missing'
    if 'subtype' in df_merged.columns:
        df_merged['subtype'] = df_merged['subtype'].replace(['', None, 'nan', 'NaN', pd.NA], 'missing')
        df_merged['subtype'] = df_merged['subtype'].fillna('missing')
    
    # Define the optional dtypes for the DataFrame columns
    opt_dtypes = {
        'ID1': 'string[pyarrow]',
        'image_laterality': 'category',
        'patient_orientation': 'category',
        'code_meaning': 'category',
        'Age': 'Int32',
        'number': 'Int32',
        'abnormality': 'string[pyarrow]',
        'classification': 'category',
        'subtype': 'category'
    }

    labels_df = pd.DataFrame()
    
    # Iterate through the optional dtypes and apply them to the DataFrame columns
    # Categorical columns are converted to 'category' dtype, which is more memory efficient for categorical data (e.g. Left/Right, MLO/CC).
    # They're however still stored as strings in the DataFrame, but with a mapping to integer codes for memory efficiency.
    for col, dtype in opt_dtypes.items():
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(dtype)
            if dtype == 'category':
                # Display the category mapping for categorical columns in the console
                print(f"\nCategory mapping for column '{col}':")
                print(dict(enumerate(df_merged[col].cat.categories)))
                # Stores the labels for the categorical data in a separate DataFrame
                labels_df = pd.concat([labels_df, df_merged[col].astype('category')], axis=1, ignore_index=True)

    # Final DataFrame
    df_cmmd = df_merged

    # Show memory usage and total number of DICOM files processed
    print(df_cmmd.info(memory_usage='deep'))
    print(f"Total number of DICOM files processed: {len(df_cmmd)}")
    
    return df_cmmd, labels_df

if __name__ == "__main__":
    print(df_cmmd.head())