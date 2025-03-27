import cv2 as cv
import numpy as np
import os

#extractes the ACF data for a single image
def acf_extract(image : cv.Mat):
    gray_tile = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Konverterer fra RGB til grayscale
    grad_x = cv.Sobel(gray_tile, cv.CV_64F, 1, 0, ksize=3) # Bruger Sobel til at beregne gradienten hen ad
    # x-aksen
    grad_y = cv.Sobel(gray_tile, cv.CV_64F, 0, 1, ksize=3) # Beregner gradient ad y-aksen

    magnitude = np.sqrt(grad_x**2 + grad_y**2) # Der udregnes magnituden for hver gradient i x- og y-matricen 
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi) # Der tages invers tangens af begge garadients
    # som sammen danner en vektor. Der ganges med 180/pi for at konvertere til grader. 
    
    # Laver et histogram for magnitude. Histogrammet er en 2D matrix, som bliver konverteret til en 1D vektor
    # (eller basically en liste)
    magnitude_hist = cv.calcHist([magnitude.astype(np.float32)], [0], None, [8], [0, 255]).flatten() 

    # Det samme som med magnitude.
    orientation_hist = cv.calcHist([orientation.astype(np.float32)], [0], None, [8], [-180, 180]).flatten()

    hsv_tile = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hue_hist = cv.calcHist([hsv_tile], [0], None, [8], [0, 180]).flatten()

    magnitude_hist /= np.sum(magnitude_hist) if np.sum(magnitude_hist) > 0 else 1
    orientation_hist /= np.sum(orientation_hist) if np.sum(orientation_hist) > 0 else 1
    hue_hist /= np.sum(hue_hist) if np.sum(hue_hist) > 0 else 1

    return (magnitude_hist, orientation_hist, hue_hist)

#extractes the ACF data for a list of images
def acf_extract_list(image_list : list[cv.Mat]):
    output = []
    for image in image_list:
        output.append(acf_extract(image))
    return output
