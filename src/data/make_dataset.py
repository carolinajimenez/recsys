import os
import shutil
import zipfile

import requests


# Get the current script's directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the folder where the files will be saved
output_folder = os.path.join(current_dir, "../../data/raw/")

# URL of the ZIP file to download
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Check if the output folder exists, and if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Name of the ZIP file
zip_filename = os.path.join(output_folder, "ml-latest-small.zip")

# Download the ZIP file
response = requests.get(url)
with open(zip_filename, "wb") as zip_file:
    zip_file.write(response.content)

# Extract the ZIP file directly into the output folder
with zipfile.ZipFile(zip_filename, "r") as zip_ref:
    zip_ref.extractall(output_folder)

# Remove the ZIP file after extraction
os.remove(zip_filename)

# Move all files from the subfolder to the output folder
extracted_folder = os.path.join(output_folder, "ml-latest-small")
for root, dirs, files in os.walk(extracted_folder):
    for file in files:
        src_file = os.path.join(root, file)
        dest_file = os.path.join(output_folder, file)
        shutil.move(src_file, dest_file)

# Remove the empty subfolder
os.rmdir(extracted_folder)

print("Download and extraction completed in folder:", output_folder)
