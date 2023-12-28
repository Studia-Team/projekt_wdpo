import json
from pathlib import Path
from typing import Dict
import click
import cv2
from tqdm import tqdm
import numpy as np

def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """

    #TODO: Implement detection method.
    red = 0
    yellow = 0
    green = 0
    purple = 0

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#green
    mask_hsv_green = cv2.inRange(hsv, (30, 190, 16), (55, 255, 255))
    kernel = np.ones((7, 7), np.uint8)
    kernel1 = np.ones((11, 11), np.uint8)
    kernel2 = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(mask_hsv_green, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    erosion = cv2.erode(closing, kernel1, iterations=1)
    dilation = cv2.dilate(closing, kernel2, iterations=1)
    cnt, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size = []
    big = False
    for c in cnt:
        size.append(cv2.contourArea(c))
    for s in size:
        if s > 8000:
            big = True

    for s1 in size:
        if big:
            if s1 > 4000:
                green += 1
        else:
            if s1 > 400:
                green += 1
#purple
    mask_hsv_purple = cv2.inRange(hsv, (145, 24, 10), (177, 255, 122))
    kernel = np.ones((5, 5), np.uint8)
    kernel1 = np.ones((11, 11), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask_hsv_purple, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
    erosion = cv2.erode(closing, kernel2, iterations=1)
    dilation = cv2.dilate(opening, kernel, iterations=1)
    cnt, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size = []
    big = False
    for c in cnt:
        size.append(cv2.contourArea(c))
    for s in size:
        if s > 3000:
            big = True

    for s1 in size:
        if big:
            if s1 > 1200:
                purple += 1
        else:
            if s1 > 600:
                purple += 1
#yellow
    mask_hsv_yellow = cv2.inRange(hsv, (6, 226, 83), (30, 255, 255))
    kernel = np.ones((7, 7), np.uint8)
    kernel1 = np.ones((9, 9), np.uint8)
    kernel2 = np.ones((11, 11), np.uint8)
    kernel3 = np.ones((8, 8), np.uint8)
    opening = cv2.morphologyEx(mask_hsv_yellow, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
    erosion = cv2.erode(closing, kernel3, iterations=1)
    dilation = cv2.dilate(erosion, kernel2, iterations=1)
    cnt, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size = []
    big = False
    for c in cnt:
        size.append(cv2.contourArea(c))
    for s in size:
        if s > 3000:
            big = True

    for s1 in size:
        if big:
            if s1 > 700:
                yellow += 1
        else:
            if s1 > 30:
                yellow += 1
#red
    mask_hsv_red = cv2.inRange(hsv, (170, 43, 125), (255, 255, 255))
    kernel = np.ones((10, 10), np.uint8)
    kernel1 = np.ones((9, 9), np.uint8)
    kernel2 = np.ones((11, 11), np.uint8)
    opening = cv2.morphologyEx(mask_hsv_red, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
    erosion = cv2.erode(closing, kernel1, iterations=1)
    dilation = cv2.dilate(erosion, kernel2, iterations=1)
    cnt, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnt:
        size = cv2.contourArea(c)
        if size > 400:
            red = red + 1

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)




if __name__ == '__main__':
    main()
