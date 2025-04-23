import cv2 as cv
import numpy as np
import os
import seaborn as sns
from TestACF import acf_extract_list
import pandas as pd
import matplotlib.pyplot as plt;
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Tile Model (Random Forest)
centerTile_path = "Classified_Tiles/CenterTile"
farmTile_path = "Classified_Tiles/Farm"
fieldTile_path = "Classified_Tiles/Field"
forestTile_path = "Classified_Tiles/Forest"
mineTile_path = "Classified_Tiles/Mines"
otherTile_path = "Classified_Tiles/Other"
swampTile_path = "Classified_Tiles/Swamp"
waterTile_path = "Classified_Tiles/Water"

Crown0Tile_path = "Classified_Tiles/0Crowns"
Crown1Tile_path = "Classified_Tiles/1Crowns"
Crown2Tile_path = "Classified_Tiles/2Crowns"
Crown3Tile_path = "Classified_Tiles/3Crowns"

ordinal_encoder_tile = OrdinalEncoder()
ordinal_encoder_crown = OrdinalEncoder()

#crowns (SIFT)
crown_template = cv.imread("King Domino dataset/CrownTemplate.png", cv.IMREAD_GRAYSCALE) #image used with sift to find crowns
sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False) #BFMatcher finds the best matches between descriptors (used for SIFT)

#main function
def main():
    crown_model = generate_crown_model()
    tile_model = generate_tile_model()
    calculate_score(cv.imread("King Domino dataset/Train/3.jpg"), tile_model)



def calculate_score(image : cv.Mat, tile_model : RandomForestClassifier):
    #processing image
    tiles = get_tiles(image)   
    acf_data = acf_extract_list(tiles)
    dataframe = pd.DataFrame(columns=["Position", "Crowns", "magnitude_hist", "orientation_hist", "hue_hist"])
    i = 0
    for acf_data_point in acf_data:
        dataframe.loc[len(dataframe)] = [(i % 5, np.floor(i / 5)), get_number_of_crowns(tiles[i]), acf_data_point[0], acf_data_point[1], acf_data_point[2]] #appends the ACF data for the tile in the dataframe
        i += 1
    
    model_input = dataframe[["magnitude_hist", "orientation_hist", "hue_hist"]]
    model_input = preprocess_x(model_input)
    model_predict = tile_model.predict(model_input)
    model_predict_labels = ordinal_encoder_tile.inverse_transform(model_predict.reshape(-1, 1)).flatten()  # Convert predictions to original labels
    
    #plot results
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(cv.cvtColor(tiles[i], cv.COLOR_BGR2RGB))
        ax.text(0.5, -0.1, f"{model_predict_labels[i]} - {dataframe["Crowns"][i]}", ha='center', va='top', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    

# Break a board into tiles (taken from p0 github)
def get_tiles(image : cv.Mat):
    tiles = []
    for y in range(5):
        for x in range(5):
            tiles.append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles    

def plot_feature_importance(model, feature_names, title):
    feature_importances = model.feature_importances_
    num_features = len(feature_importances)

    # Ensure the feature names match the number of columns
    if len(feature_names) != num_features:
        feature_names = [f"Feature {i}" for i in range(num_features)]  # Dynamically name features

    indices = np.argsort(feature_importances)[::-1]  # Sort in descending order
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[feature_names[i] for i in indices], y=feature_importances[indices])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def generate_crown_model():
    data = pd.DataFrame(columns=["TileType", "magnitude_hist", "orientation_hist", "hue_hist"])
    extract_and_save_ACF_data(Crown0Tile_path, data, "0")
    extract_and_save_ACF_data(Crown1Tile_path, data, "1")
    extract_and_save_ACF_data(Crown2Tile_path, data, "2")
    extract_and_save_ACF_data(Crown3Tile_path, data, "3")
    print("Done Extracting Data")
    
    train_x, test_x, train_y, test_y = train_test_split(
        data[["magnitude_hist", "orientation_hist", "hue_hist"]],
        data[["TileType"]],
        test_size=0.2,
        random_state=42
    )
    
    ordinal_encoder_crown.fit(data[["TileType"]])
    train_x = preprocess_x(train_x)
    train_y = preprocess_y(train_y, ordinal_encoder_crown)
    
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    
    plot_feature_importance(model, ["Magnitude Histogram", "Orientation Histogram", "Hue Histogram"], "Crown Model Feature Importance")
    return model

def generate_tile_model():
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
    
    train_x, test_x, train_y, test_y = train_test_split(
        data[["magnitude_hist", "orientation_hist", "hue_hist"]],
        data[["TileType"]],
        test_size=0.2,
        random_state=42
    )
    
    ordinal_encoder_tile.fit(data[["TileType"]])
    train_x = preprocess_x(train_x)
    train_y = preprocess_y(train_y, ordinal_encoder_tile)
    
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    
    plot_feature_importance(model, ["Magnitude Histogram", "Orientation Histogram", "Hue Histogram"], "Tile Model Feature Importance")
    return model


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
    magnitude_hist_frame = dataframe["magnitude_hist"].apply(lambda x: np.array(x))
    orientation_hist_frame = dataframe["orientation_hist"].apply(lambda x: np.array(x))
    hue_hist_frame = dataframe["hue_hist"].apply(lambda x: np.array(x))
    
    # Convert lists to individual columns
    dataframe_expanded = np.hstack([
        np.vstack(magnitude_hist_frame),
        np.vstack(orientation_hist_frame),
        np.vstack(hue_hist_frame)
    ])

    return pd.DataFrame(dataframe_expanded)  # Return the new DataFrame

def preprocess_y(dataframe : pd.DataFrame, encoder : OrdinalEncoder):
    dataframe["TileType"] = encoder.transform(dataframe)
    return dataframe["TileType"]

#gets the number of crowns on a tile
def get_number_of_crowns(image : cv.Mat):
    target = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Find keypoints and descriptors in the template image and target image
    kp_template, des_template = sift.detectAndCompute(crown_template, None)
    kp_target, des_target = sift.detectAndCompute(target, None)
    
    # Match the descriptors using KNN
    matches = bf.knnMatch(des_template, des_target, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m in matches:
        if len(m) == 2:  # Ensure there are two matches
            m, n = m  # Unpack the two nearest neighbors
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        elif len(m) == 1:  # Handle the case where there is only one match
            good_matches.append(m[0])  # Directly append the single match
    
    return len(good_matches)

if __name__ == "__main__":
    main()