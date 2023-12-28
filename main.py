
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image():
    # Load the image
    image_path = 'G:\\001_STUDIA\\002_SEM_5\\001_WDPO\projekt\\data\\0003.jpg'
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform thresholding to create a binary image
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Display the images using matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image')

    plt.show()




def main():
    read_image()


if __name__ == '__main__':
    main()


