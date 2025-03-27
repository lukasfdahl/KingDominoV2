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
from sklearn.metrics import classification_report

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
    tile_model = generate_tile_model()
    calculate_score(cv.imread("King Domino dataset/Train/3.jpg"), tile_model)



def calculate_score(image : cv.Mat, tile_model : RandomForestClassifier):
    #processing image
    tiles = get_tiles(image)   
    acf_data = acf_extract_list(tiles)
    dataframe = pd.DataFrame(columns=["Position", "magnitude_hist", "orientation_hist", "hue_hist"])
    i = 0
    for acf_data_point in acf_data:
        dataframe.loc[len(dataframe)] = [(i % 5, np.floor(i / 5)), acf_data_point[0], acf_data_point[1], acf_data_point[2]] #appends the ACF data for the tile in the dataframe
        i += 1
    
    model_input = preprocess_x(dataframe[["magnitude_hist", "orientation_hist", "hue_hist"]])
    model_predict = tile_model.predict(model_input)
    model_predict_labels = ordinal_encoder.inverse_transform(model_predict.reshape(-1, 1)).flatten()  # Convert predictions to original labels
    
    #plot results
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(cv.cvtColor(tiles[i], cv.COLOR_BGR2RGB))
        ax.text(0.5, -0.1, model_predict_labels[i], ha='center', va='top', transform=ax.transAxes, fontsize=10)
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

#collects data, preprocesses data, and traines model
def generate_tile_model():
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
    
    #train model
    print("Training model")
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    
    ''' For generating performance repport and confusion matrix
    #test model
    print("testing model")
    test_x = preprocess_x(test_x)
    test_y = preprocess_y(test_y)
    pred_y = model.predict(test_x)
    
    pred_y_labels = ordinal_encoder.inverse_transform(pred_y.reshape(-1, 1)).flatten()  # Convert predictions to original labels
    test_y_labels = ordinal_encoder.inverse_transform(test_y.values.reshape(-1, 1)).flatten()  # Convert actual values to original labels
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_y_labels, pred_y_labels, labels=ordinal_encoder.categories_[0])  # Ensure categories are used
    
    print(classification_report(test_y_labels, pred_y_labels, labels=ordinal_encoder.categories_[0]))
    
    # Display the confusion matrix with actual labels
    plt.figure(figsize=(10,7))
    sn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=ordinal_encoder.categories_[0], yticklabels=ordinal_encoder.categories_[0])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    '''
    print("Done Traning model")
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

#gets the number of crowns on a tile
def get_number_of_crowns(image : cv.Mat):
    pass

if __name__ == "__main__":
    main()