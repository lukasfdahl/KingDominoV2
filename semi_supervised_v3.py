import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ensure output directory exists
output_dir = "Classified_Tiles"
os.makedirs(output_dir, exist_ok=True)

tileData = []  # Stores feature vectors
tilePosition = []  # Stores tile positions
tileImages = []  # Stores tile images

def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")

    for file_number in range(1, 75):
        image_path = f"King Domino dataset/Train/{file_number}.jpg"

        if not os.path.isfile(image_path):
            print(f"Image {file_number}.jpg not found")
            continue

        image = cv.imread(image_path)
        tiles = get_tiles(image)

        print(f"Total tiles extracted: {len(tiles) * len(tiles[0])}\n")

        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                get_terrain(tile, x, y, file_number)

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
    print(f"standardized_tileData = {standardized_tileData}")

    # Apply KMeans clustering on standardized features
    kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
    kmeans.fit(standardized_tileData)

    categoryDictionary = {}
    for i, tile in enumerate(standardized_tileData):
        category = kmeans.predict(tile.reshape(1, -1))[0]
        if category not in categoryDictionary:
            categoryDictionary[category] = []
        categoryDictionary[category].append((tilePosition[i], tileImages[i]))

    # Create folders and save tiles
    for category, tiles in categoryDictionary.items():
        category_folder = os.path.join(output_dir, f"Category_{category}")
        os.makedirs(category_folder, exist_ok=True)

        for (file_number, x, y), tile in tiles:
            tile_filename = f"tile_{file_number}_{x}_{y}.jpg"
            tile_path = os.path.join(category_folder, tile_filename)
            cv.imwrite(tile_path, tile)
            
        print(f"Category {category}: {len(tiles)} tiles saved in {category_folder}")

# Break a board into 5x5 tiles
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tile = image[y * 100:(y + 1) * 100, x * 100:(x + 1) * 100]
            tiles[-1].append(tile)
    return tiles

# Extract features from a tile and store its position and image
def get_terrain(tile, x, y, file_number):
    gray_tile = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray_tile, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray_tile, cv.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi)

    magnitude_hist = cv.calcHist([magnitude.astype(np.float32)], [0], None, [8], [0, 255]).flatten()
    orientation_hist = cv.calcHist([orientation.astype(np.float32)], [0], None, [8], [-180, 180]).flatten()
    
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue_hist = cv.calcHist([hsv_tile], [0], None, [8], [0, 180]).flatten()

    magnitude_hist /= np.sum(magnitude_hist) if np.sum(magnitude_hist) > 0 else 1
    orientation_hist /= np.sum(orientation_hist) if np.sum(orientation_hist) > 0 else 1
    hue_hist /= np.sum(hue_hist) if np.sum(hue_hist) > 0 else 1

    feature_vector = np.concatenate((magnitude_hist, orientation_hist, hue_hist))
    
    tileData.append(feature_vector)
    tilePosition.append((file_number, x, y))
    tileImages.append(tile)

if __name__ == "__main__":
    main()
