import matplotlib.pyplot as plt
import cv2
import pandas as pd


from skimage.feature import hog
from skimage import exposure

image = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg", cv2.IMREAD_GRAYSCALE)
print(image.shape)

fd, hog_image = hog(
    image,
    orientations=8,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    visualize=True
)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()