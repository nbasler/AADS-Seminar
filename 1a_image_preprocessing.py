import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler

# Preprocessing methods for contour detection and cropping images_________________________________

def find_largest_bbox(image_array):
    """
    Detects the Region of Interest (ROI) using cv2.findContours in the provided pixel array
    and returns its bounding box (x, y, w, h), i.e. bbox, in the image's own coordinate system.
    
    Args:
        - image_array (color or grayscale): The image in which to find the ROI.
    Returns:
        - x, y, w, h: Starting point on x-axis, on y-axis, width and height
        - None: If no contours are found or image is unsuitable.
    """
    
    if image_array is None:
        return None

    # Ensure the image is in grayscale format
    if len(image_array.shape) == 3:
        # Convert to grayscale if the image is in color (3 channels)
        processed_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    elif len(image_array.shape) == 2:
        # If already in grayscale, normalize it to ensure pixel values are in the range [0, 255]
        processed_image = cv2.normalize(image_array.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif len(image_array.shape) < 2:
        print("Warning: Image array does not have enough dimensions for contour detection.")
        return None
    
    # ensure image is in np.uint8 format
    if processed_image.dtype != np.uint8:
        processed_image = processed_image.astype(np.uint8) 
    
    if processed_image is None or processed_image.size == 0:
        return None

    # Using OTSU thresholding to find automatically thresholds to then create a binary image,
    # which sets all pixels above the threshold to 255 (white) and below to 0 (black).
    _, thresh_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # '_' is used to neglect the threshold value returned by cv2.threshold
    
    # Morphological operations to remove noise and fill holes we use a kernel of ones (3x3),
    # which is a small square that will be used to process the binary image.
    # This helps to clean up the binary image by removing small noise and closing small holes.
    kernel = np.ones((3, 3), np.uint8)
    # MORPH_OPEN removes small noise by first eroding and then dilating the image.
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)
    # MORPH_Close closes holes by first dilating and then eroding the image, i.e. inverse of MORPH_OPEN.
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

    # Find contours including internal ones (i.e. boxes within boxes that thus are lower in the hierarchy, see openCV documentation).
    # cv2.RETR_TREE retrieves all contours and reconstructs a full hierarchy of nested contours, i.e. Tree.
    # Opposing to cv2.RETR_EXTERNAL, which retrieves only the external contours.
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # '_' is used to ignore the hierarchy output
    if not contours:
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h

def find_common_bbox(image_list, lateralities_list):
    """
    This function processes each image to find the bounding box of the largest contour in all images,
    and returns a list of bounding box parameters (x_start, y_start, x_end, y_end) that can be used to crop all images
    to a common size while considering the laterality of each image.
    
    Args:
        - image_list (list): List of images (numpy arrays).
        - lateralities_list (list): List of lateralities ('L' or 'R') corresponding to each image.
    Returns:
        - bbox_params: list with common_x, common_y, common_width, common_height, or None.
    """
    bboxes_params = [] # Stores (x_start, y_start, x_end, y_end)

    for img, lat in zip(image_list, lateralities_list):
        if img is None or img.size == 0:
            continue
        # Ensure image is 2D for shape attribute
        current_img_shape = img.shape
        if len(current_img_shape) < 2:
            print(f"Warning: Skipping image with unexpected shape {current_img_shape} in find_common_bbox.")
            continue
        
        bbox_local = find_largest_bbox(img) # Gets (x, y, w, h) in image's own coords
        bboxes_params.append((bbox_local)) # Append the bounding box parameters (x, y, w, h)

    if not bboxes_params:
        return None
    # Return the list of individual bounding boxes parameters for given list of images
    return bboxes_params

def crop_images_bbox(image_list, lateralities_list):
    """
    Crops all images in the list to the common ROI bounding box,
    while adjusting for laterality. 'R' means the right side of the image,
    and 'L' means the left side of the image. For simplicity the images arre not cropped at the height, but at the width,
    as this would further complicate the cropping process (i.e. aligning the bbox per image).
    This allows us to crop the images width, starting from the left for 'L' and from the right for 'R'.
    
    Args:
        - image_list (list): List of images (numpy arrays).
        - lateralities_list (list): List of lateralities ('L' or 'R') corresponding to each image (string).
    Returns:
        - cropped_images (list): List of cropped images or None for failed crops.
        - common_w (int): Common width of the bounding box across all images.
        - common_h (int): Common height of the bounding box across all images.
    """
    
    if not image_list:
        return []

    bboxes_params = find_common_bbox(image_list, lateralities_list)
    if not bboxes_params:
        print("No valid bounding boxes found. Returning empty list.")
        return []
    
    # Extract widths from all bounding boxes and find maximum
    common_x_left = 0 # Starting x-coordinate for left side cropping
    common_x_right = image_list[0].shape[1] # Starting x-coordinate for right side cropping
    common_y = 0 # Starting y-coordinate for cropping, i.e. top of the image
    common_h = image_list[0].shape[0] # Height of the image
    widths = [bbox[2] for bbox in bboxes_params]  # bbox[2] is width
    common_w = max(widths) # max width across all bounding boxes
    print(f'Common bounding box dimensions: w: {common_w}, h: {common_h}')
    
    cropped_images = []
    
    # Loop through all images and crop them to the common bounding box, taking into account the laterality
    for img, lat in zip(image_list, lateralities_list):
        # Check if the image is valid
        if img is None or img.size == 0:
            cropped_images.append(None)
            continue
        
        # Raise warning if image is not 2D, image then is not cropped
        current_img_shape = img.shape
        if len(current_img_shape) < 2:
            print(f"Warning: Skipping image with unexpected shape {current_img_shape} in crop_images_to_common_roi_cv2.")
            cropped_images.append(None)
            continue
        
        # Determine actual crop coordinates for the current image, which depends on the laterality
        if lat == 'R':
            x_crop_start = common_x_right - common_w
            x_crop_end = common_x_right
        elif lat == 'L':
            x_crop_start = common_x_left
            x_crop_end = common_x_left + common_w

        # Perform the crop
        cropped_img = img[common_y:common_h, x_crop_start:x_crop_end]

        cropped_images.append(cropped_img)
        
    return cropped_images, common_w, common_h