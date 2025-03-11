import cv2 as cv
import numpy as np
import os
from ACF import acf_extract_list
import pandas as pd

centerTile_path = "Classified_Tiles/CenterTile"


def main():
    data = pd.DataFrame(columns=["TileType", "magnitude_hist", "orientation_hist", "hue_hist"])
    center_values = extract_image_data(centerTile_path)
    for value in center_values:
       data.loc[len(data)] = ["Center", value[0], value[1], value[2]]
    print(data)



def extract_image_data(folder_path):
    files_in_folder = os.listdir(folder_path)
    billed_data = []
    for file in files_in_folder:
        image_path = f"{centerTile_path}/{file}"

        billed_data.append(cv.imread(image_path))

    return acf_extract_list(billed_data)






if __name__ == "__main__":
    main()