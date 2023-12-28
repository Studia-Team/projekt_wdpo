import cv2
import numpy as np
import os
from scipy.spatial import procrustes

def load_and_preprocess_image(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, 0)
    # Apply a blur
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Binarize the image
    _, binary_image = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image

def find_largest_contour(binary_image):
    # Find all contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Return the largest contour
    return max(contours, key=cv2.contourArea)

def normalize_contour(contour, size=(100, 100)):
    # Ensure the contour is in the correct shape (n, 1, 2)
    if contour.shape[-1] != 2 or len(contour.shape) != 3:
        contour = contour.reshape(-1, 1, 2)

    # Compute the bounding box of the contour and then use it to compute the aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    # Compute the new width and height based on the aspect ratio
    if aspect_ratio > 1:
        # Width is greater than height, so width should be the larger dimension
        new_w = size[0]
        new_h = int(size[0] / aspect_ratio)
    else:
        # Height is greater than width, so height should be the larger dimension
        new_h = size[1]
        new_w = int(size[1] * aspect_ratio)

    # Compute the resize scale for both dimensions
    scale_x = new_w / w
    scale_y = new_h / h

    # Translate the contour to the origin and then scale
    normalized_contour = (contour - [x, y]) * [scale_x, scale_y]

    return normalized_contour.astype(np.int32)

def average_contours(contours):
    # This function averages a list of contours
    # We need to ensure that all contours have the same number of points
    # As an example, we're just going to take the first contour and resize the rest to match its size
    standard_length = len(contours[0])
    resized_contours = [cv2.resize(contour, (standard_length, 1), interpolation=cv2.INTER_LINEAR) for contour in contours]
    # Now we can simply take the mean of all resized contours
    mean_contour = np.mean(np.array(resized_contours), axis=0).astype(contours[0].dtype)
    return mean_contour

def process_leaf_images(folder_path):
    all_contours = []
    # Iterate over all images in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Full path to the image
            image_path = os.path.join(folder_path, filename)
            # Preprocess the image
            binary_image = load_and_preprocess_image(image_path)
            # Find the largest contour
            largest_contour = find_largest_contour(binary_image)
            # Normalize the contour
            normalized_contour = normalize_contour(largest_contour)
            all_contours.append(normalized_contour)
    
    # Average the contours
    average_contour = average_contours(all_contours)
    return average_contour

# Example usage
# Replace 'path_to_leaf_folder' with the actual folder path containing your leaf images
representative_contour = process_leaf_images('G:\\001_STUDIA\\002_SEM_5\\001_WDPO\\projekt\\mapple_jpg')

