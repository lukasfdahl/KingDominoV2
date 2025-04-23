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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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

tile_map : dict[tuple[int, int], tuple[str, int]] = {}

#main function
def main():
    crown_model = generate_crown_model()
    tile_model = generate_tile_model()
    
    print("doen making models")
    
    #run on test set
    pre_calculated_scores = pd.read_csv("kingdomino_scores.csv")

    #get images and names from test set
    files_in_test_folder = os.listdir("King Domino dataset/Test")
    y_true = []
    y_pred = []
    
    for file in files_in_test_folder:
        predicted_score = calculate_score(cv.imread(f"King Domino dataset/Test/{file}"), tile_model, crown_model)
        y_pred.append(predicted_score)
        board_number = file.removesuffix(".jpg")
        true_score = pre_calculated_scores[pre_calculated_scores["TileNumber"] == int(board_number)]["Score"].values[0]
        print(f"true score = {true_score}, predicted score = {predicted_score}")
        y_true.append(true_score)
        
    full_acc_num = 0
    for index, element in enumerate(y_true):
        if element == y_pred[index]:
            full_acc_num += 1
            
    total_board_num = len(y_pred)
    
    print(f"precentage of boards with 100% accurate predections = {full_acc_num / total_board_num}")
    print(f"mean squared error = {mean_squared_error(y_true, y_pred)}")
    print(f"mean absolute error = {mean_absolute_error(y_true, y_pred)}")
    
    abs_errors = np.abs(np.array(y_true) - np.array(y_pred))
    median_abs_error = np.median(abs_errors)

    print(f"median absolute error = {median_abs_error:.2f}")
    
    #######################
    # draw plot ###########
    #######################
    
    board_indices = list(range(len(y_true)))
    abs_error = [abs(t - p) for t, p in zip(y_true, y_pred)]

    fig, ax1 = plt.subplots(figsize=(15, 6))

    width = 0.4
    ax1.bar([i - width/2 for i in board_indices], y_true, width=width, label='True Score', color='skyblue')
    ax1.bar([i + width/2 for i in board_indices], y_pred, width=width, label='Predicted Score', color='orange')
    ax1.set_xlabel('Board Index')
    ax1.set_ylabel('Score')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    # Secondary y-axis for absolute error

    # Annotate number of perfect predictions
    perfect_preds = sum([t == p for t, p in zip(y_true, y_pred)])
    plt.title(f'Score Predictions vs True Scores (Perfect predictions: {perfect_preds}/{len(y_true)})')

    plt.tight_layout()
    plt.show()
        

def calculate_score(image : cv.Mat, tile_model : RandomForestClassifier, crown_model : RandomForestClassifier):
    #processing image
    tiles = get_tiles(image)   
    acf_data = acf_extract_list(tiles)
    dataframe = pd.DataFrame(columns=["Position", "Crowns", "magnitude_hist", "orientation_hist", "hue_hist"])
    i = 0
    for acf_data_point in acf_data:
        dataframe.loc[len(dataframe)] = [(i % 5, np.floor(i / 5)), i, acf_data_point[0], acf_data_point[1], acf_data_point[2]] #appends the ACF data for the tile in the dataframe
        i += 1
    
    model_input = dataframe[["magnitude_hist", "orientation_hist", "hue_hist"]]
    model_input = preprocess_x(model_input)
    dataframe["Crowns"] = dataframe["Crowns"].apply(lambda x: get_number_of_crowns(model_input.iloc[x], crown_model))
    model_predict = tile_model.predict(model_input)
    model_predict_labels = ordinal_encoder_tile.inverse_transform(model_predict.reshape(-1, 1)).flatten()  # Convert predictions to original labels

    # Loop over each row in the dataframe to build the tile map
    for i, acf_data_point in enumerate(acf_data):
        # Get the position as a tuple (x, y)
        position = (i % 5, int(np.floor(i / 5)))
        
        # Get the predicted tile type from the model prediction labels
        predicted_tile_type = model_predict_labels[i]
        
        # Get the number of crowns from the dataframe
        crowns = int(dataframe.at[i, "Crowns"][0])
        
        # Store the (predicted_tile_type, crowns) tuple for the given position in tile_map
        tile_map[position] = (predicted_tile_type, crowns)
    score = 0
    while len(tile_map) > 0:
        first_key, first_value = next(iter(tile_map.items()))
        crowns_num, tile_num = calc_score_block(first_key, tile_map[first_key][0])
        score += crowns_num * tile_num

    return score
    

# Break a board into tiles (taken from p0 github)
def get_tiles(image : cv.Mat):
    tiles = []
    for y in range(5):
        for x in range(5):
            tiles.append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles    

#collects data, preprocesses data, and traines model
def generate_crown_model():
    #extract traning data and save to dataframe
    data = pd.DataFrame(columns=["TileType", "magnitude_hist", "orientation_hist", "hue_hist"])
    extract_and_save_ACF_data(Crown0Tile_path, data, "0")
    extract_and_save_ACF_data(Crown1Tile_path, data, "1")
    extract_and_save_ACF_data(Crown2Tile_path, data, "2")
    extract_and_save_ACF_data(Crown3Tile_path, data, "3")
    print("Done Extracting Data")
    
    #test train split
    train_x, test_x, train_y, test_y = train_test_split(data[["magnitude_hist", "orientation_hist", "hue_hist"]], data[["TileType"]], test_size = 0.2, random_state = 42)
    
    #preprocess
    ordinal_encoder_crown.fit(data[["TileType"]])
    train_x = preprocess_x(train_x)
    train_y = preprocess_y(train_y, ordinal_encoder_crown)
    
    #train model
    print("Training model")
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    
    '''#For generating performance repport and confusion matrix
    #test model
    print("testing model")
    test_x = preprocess_x(test_x)
    test_y = preprocess_y(test_y, ordinal_encoder_crown)
    pred_y = model.predict(test_x)
    
    pred_y_labels = ordinal_encoder_crown.inverse_transform(pred_y.reshape(-1, 1)).flatten()  # Convert predictions to original labels
    test_y_labels = ordinal_encoder_crown.inverse_transform(test_y.values.reshape(-1, 1)).flatten()  # Convert actual values to original labels
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_y_labels, pred_y_labels, labels=ordinal_encoder_crown.categories_[0])  # Ensure categories are used
    
    print(classification_report(test_y_labels, pred_y_labels, labels=ordinal_encoder_crown.categories_[0]))
    
    # Display the confusion matrix with actual labels
    plt.figure(figsize=(10,7))
    sn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=ordinal_encoder_crown.categories_[0], yticklabels=ordinal_encoder_crown.categories_[0])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()'''
    
    print("Done Traning model")
    return model

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
    ordinal_encoder_tile.fit(data[["TileType"]])
    train_x = preprocess_x(train_x)
    train_y = preprocess_y(train_y, ordinal_encoder_tile)
    
    #train model
    print("Training model")
    model = RandomForestClassifier()
    model.fit(train_x, train_y)

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
def get_number_of_crowns(acf, crown_model : RandomForestClassifier):
    return ordinal_encoder_crown.inverse_transform(crown_model.predict([acf]).reshape(-1, 1)).flatten()


def calc_score_block(position : tuple[int, int], tile_type : str):
    neighbour_positions = [(position[0]-1, position[1]),  # left
                       (position[0]+1, position[1]),  # right   
                       (position[0], position[1]-1),  # up
                       (position[0], position[1]+1)]  # down
    crown_num = tile_map[position][1]
    number_of_tiles = 1
    tile_map.pop(position)

    for neighbour_position in neighbour_positions:
        if neighbour_position in tile_map and tile_map[neighbour_position][0] == tile_type:
            neighbour_crown_num, neighbour_tile_num = calc_score_block(neighbour_position, tile_type)
            crown_num += neighbour_crown_num
            number_of_tiles += neighbour_tile_num

    return crown_num, number_of_tiles

if __name__ == "__main__":
    main()