import cv2 as cv
import numpy as np
import os
from TestACF import acf_extract_list
import pandas as pd
import matplotlib.pyplot as plt;
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix

centerTile_path = "Classified_Tiles/CenterTile"
farmTile_path = "Classified_Tiles/Farm"
fieldTile_path = "Classified_Tiles/Field"
forestTile_path = "Classified_Tiles/Forest"
mineTile_path = "Classified_Tiles/Mines"
otherTile_path = "Classified_Tiles/Other"
swampTile_path = "Classified_Tiles/Swamp"
waterTile_path = "Classified_Tiles/Water"

ordinal_encoder = OrdinalEncoder()

#main function
def main():
    #extract traning data and save to dataframe
    data = pd.DataFrame(columns=["TileType", "magnitude_hist", "orientation_hist", "hue_hist"])
    extract_and_save_ACF_data(centerTile_path, data, "Center")
    extract_and_save_ACF_data(farmTile_path, data, "Farm")
    extract_and_save_ACF_data(fieldTile_path, data, "Field")
    extract_and_save_ACF_data(forestTile_path, data, "Forest")
    extract_and_save_ACF_data(mineTile_path, data, "Mine")
    extract_and_save_ACF_data(otherTile_path, data, "Other")
    extract_and_save_ACF_data(swampTile_path, data, "Swamp")
    extract_and_save_ACF_data(waterTile_path, data, "Water")
    print("Done Extracting Data")
    
    #test train split
    train_x, test_x, train_y, test_y = train_test_split(data[["magnitude_hist", "orientation_hist", "hue_hist"]], data[["TileType"]], test_size = 0.2, random_state = 42)
    
    #preprocess
    ordinal_encoder.fit(data[["TileType"]])
    train_x = preprocess_x(train_x)
    train_y = preprocess_y(train_y)
    
    print(train_x)
    print(train_y)
    #train model
    print("Training model")
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    
    #test model
    print("testing model")
    test_x = preprocess_x(test_x)
    test_y = preprocess_y(test_y)
    
    pred_y = model.apply(test_x)
    conf_matrix = confusion_matrix(test_y, pred_y)
    sn.heatmap(conf_matrix)
    plt.show()


#extraqctes all image data from a specefic folder
def extract_and_save_ACF_data(folder_path, dataframe, tile_name):
    files_in_folder = os.listdir(folder_path)
    billed_data = []
    for file in files_in_folder:
        image_path = f"{folder_path}/{file}"
        billed_data.append(cv.imread(image_path))

    acf_data = acf_extract_list(billed_data)
    
    for acf_data_point in acf_data:
        dataframe.loc[len(dataframe)] = [tile_name, acf_data_point[0], acf_data_point[1], acf_data_point[2]] #appends the ACF data for the tile in the dataframe


def preprocess_x(dataframe : pd.DataFrame):
    dataframe["magnitude_hist"] = dataframe["magnitude_hist"].apply(lambda x: np.array(x))
    dataframe["orientation_hist"] = dataframe["orientation_hist"].apply(lambda x: np.array(x))
    dataframe["hue_hist"] = dataframe["hue_hist"].apply(lambda x: np.array(x))
    
    # Convert lists to individual columns
    dataframe_expanded = np.hstack([
        np.vstack(dataframe["magnitude_hist"]),
        np.vstack(dataframe["orientation_hist"]),
        np.vstack(dataframe["hue_hist"])
    ])

    return pd.DataFrame(dataframe_expanded)  # Return the new DataFrame

def preprocess_y(dataframe : pd.DataFrame):
    dataframe["TileType"] = ordinal_encoder.transform(dataframe)
    return dataframe["TileType"]




if __name__ == "__main__":
    main()