import cv2 as cv
import numpy as np
import os

# Main function containing the backbone of the program
def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")
    image_path = r"King Domino dataset\Cropped and perspective corrected boards\1.jpg"
    if not os.path.isfile(image_path):
        print("Image not found")
        return
    image = cv.imread(image_path)
    tiles = get_tiles(image)
    print(len(tiles))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            print(f"Tile ({x}, {y}):")
            print(get_terrain(tile, x, y))
            print("=====")

# Break a board into tiles
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

# Determine the type of terrain in a tile
def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    


if __name__ == "__main__": 
    main()



from sklearn.cluster import KMeans

tileData = []
tilePosition = []

kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
kmeans.fit(tileData)

categoryDictioanry = {}

i = 0
for tile in tileData:
    closest_center = kmeans.predict(tile)
    if closest_center in categoryDictioanry:
        categoryDictioanry[closest_center].append(tilePosition[i])
    else:
        categoryDictioanry[closest_center] = []
        categoryDictioanry[closest_center].append(tilePosition[i])
    i += 1


print(categoryDictioanry)