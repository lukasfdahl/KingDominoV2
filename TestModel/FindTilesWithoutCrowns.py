import os
import shutil
import hashlib

# Define folder paths
main_folders = ["Classified_Tiles/1Crowns", "Classified_Tiles/2Crowns", "Classified_Tiles/3Crowns"]
content_folders = ["Classified_Tiles/CenterTile", "Classified_Tiles/Farm", "Classified_Tiles/Field", "Classified_Tiles/Forest",
                   "Classified_Tiles/Mines", "Classified_Tiles/Other", "Classified_Tiles/Swamp", "Classified_Tiles/Water"]
destination_folder = "Classified_Tiles/0Crowns"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

def get_image_hash(image_path):
    """Generate a hash for an image file to check uniqueness."""
    hasher = hashlib.md5()
    with open(image_path, 'rb') as img:
        hasher.update(img.read())
    return hasher.hexdigest()

# Collect hashes of images in main folders
existing_hashes = set()
for folder in main_folders:
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                existing_hashes.add(get_image_hash(file_path))

# Process content folders and copy unique images
for folder in content_folders:
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                file_hash = get_image_hash(file_path)

                if file_hash not in existing_hashes:
                    shutil.copy(file_path, os.path.join(destination_folder, file))
                    existing_hashes.add(file_hash)  # Add new file hash to avoid duplication

print("Process completed! Unique images copied to:", destination_folder)