import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

tileData = []  # Stores feature vectors
tilePosition = []  # Stores tile positions

# Main function containing the backbone of the program
def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")


    image_path = r"C:\Kodning\P2\Train\2.jpg"

    if not os.path.isfile(image_path):
        print("Image not found")
        return
    
    image = cv.imread(image_path)
    tiles = get_tiles(image)

    print(f"Total tiles extracted: {len(tiles) * len(tiles[0])}\n")

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            get_terrain(tile, x, y)

    # Convert to numpy array
    tileData_np = np.array(tileData)

    # Split features (assuming 8 bins per histogram)
    magnitude_features = tileData_np[:, :8]
    orientation_features = tileData_np[:, 8:16]
    hue_features = tileData_np[:, 16:]

    # Standardize each feature group separately
    scaler_magnitude = StandardScaler()
    scaler_orientation = StandardScaler()
    scaler_hue = StandardScaler()

    magnitude_scaled = scaler_magnitude.fit_transform(magnitude_features)
    orientation_scaled = scaler_orientation.fit_transform(orientation_features)
    hue_scaled = scaler_hue.fit_transform(hue_features)

    # Concatenate the separately scaled features
    standardized_tileData = np.hstack((magnitude_scaled, orientation_scaled, hue_scaled))
    

    # Apply KMeans clustering on standardized features
    kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
    kmeans.fit(standardized_tileData)


    categoryDictioanry = {}
    i = 0
    for tile in standardized_tileData:
        closest_center = kmeans.predict(tile.reshape(1, -1))[0]
        if closest_center in categoryDictioanry:
            categoryDictioanry[closest_center].append(tilePosition[i])
        else:
            categoryDictioanry[closest_center] = []
            categoryDictioanry[closest_center].append(tilePosition[i])
        i += 1

    i2 = 0
    for category in categoryDictioanry.keys():
        print(f"category {i2} = {categoryDictioanry[category]}")
        i2 += 1

# Break a board into 5x5 tiles
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

# Extract features from a tile using ACF and HSV histograms
def get_terrain(tile, x, y):
    # Convert to grayscale for gradient features
    gray_tile = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)

    # Compute Sobel gradients
    grad_x = cv.Sobel(gray_tile, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray_tile, cv.CV_64F, 0, 1, ksize=3)

    # Compute magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Convert to degrees

    # Compute histograms
    magnitude_hist = cv.calcHist([magnitude.astype(np.float32)], [0], None, [8], [0, 255]).flatten()
    orientation_hist = cv.calcHist([orientation.astype(np.float32)], [0], None, [8], [-180, 180]).flatten()

    # Convert to HSV and compute hue histogram
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue_hist = cv.calcHist([hsv_tile], [0], None, [8], [0, 180]).flatten()

    # Find dominant hue bin
    dominant_hue_bin = np.argmax(hue_hist)

    # Normalize histograms
    magnitude_hist /= np.sum(magnitude_hist) if np.sum(magnitude_hist) > 0 else 1
    orientation_hist /= np.sum(orientation_hist) if np.sum(orientation_hist) > 0 else 1
    hue_hist /= np.sum(hue_hist) if np.sum(hue_hist) > 0 else 1

    # Create feature vector
    feature_vector = np.concatenate((magnitude_hist, orientation_hist, hue_hist))
    
    tileData.append(feature_vector)
    tilePosition.append([x, y])

    

if __name__ == "__main__":
    main()
