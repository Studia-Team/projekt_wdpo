import cv2
import numpy as np
import os

def create_reference_contour(leaf_jpg_folder, leaf_npy_folder):
    # Initialize an empty array to store the contours
    contours_list = []

    # Loop through every image in the folder
    for filename in os.listdir(leaf_jpg_folder):
        if filename.endswith('.jpg'):  # or '.png' or whatever file type your images are
            image_path = os.path.join(leaf_jpg_folder, filename)
            image = cv2.imread(image_path, 0)
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
            _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # We assume the largest contour is the leaf contour
            largest_contour = max(contours, key=cv2.contourArea)
            contours_list.append(largest_contour)

    # Now you can process the contours_list to create a single reference contour for this leaf type.
    # This could be as simple as taking the average contour, or more complex like a consensus shape.
    # For simplicity, let's just take the first contour as the reference for now.
    reference_contour = contours_list[0]

    # Save the contour data
    np.save(os.path.join(leaf_npy_folder, 'reference_contour.npy'), reference_contour)

    return reference_contour

# Example usage:
# replace 'path/to/leaf_type_folder' with the actual path to your folder of leaf images
# create_reference_contour('path/to/leaf_type_folder')


def main():
    create_reference_contour('G:\\001_STUDIA\\002_SEM_5\\001_WDPO\\projekt\\mapple_npy','G:\\001_STUDIA\\002_SEM_5\\001_WDPO\\projekt\\mapple_npy')
    

if __name__ == '__main__':
    main()