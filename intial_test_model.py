import matplotlib.pyplot as plt
import cv2
import pandas as pd


from skimage.feature import hog
from skimage import exposure

image = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg", cv2.IMREAD_GRAYSCALE)

fd, hog_image = hog(
    image,
    orientations=2,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    visualize=True,
    feature_vector=True
)

print(fd.shape)

