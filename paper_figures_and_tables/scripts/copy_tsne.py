import pickle
import os

# Define the paths to the two folders
folder1 = "cache/datasets.tsne"
folder2 = "cache/datasets.backup"

# Define the output folder
output_folder = "cache/datasets"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through the files in folder1
for filename in os.listdir(folder1):
    # Check if the file is a pickle file
    if filename.endswith(".pickle"):
        # Load the pickle file from folder1
        with open(os.path.join(folder1, filename), "rb") as f:
            data1 = pickle.load(f)

        # Load the corresponding pickle file from folder2
        with open(os.path.join(folder2, filename), "rb") as f:
            data2 = pickle.load(f)

        # Extract the TSNE embeddings from data1
        tsne = data1["X2D"]

        # Combine the TSNE embeddings with the rest of the data from data2
        combined_data = {
            "dataset": data2["dataset"],
            "X2D": tsne,
            "dataset_name": data2["dataset_name"],
            "dataset_info": data2["dataset_info"],
        }

        # Save the combined data to a new pickle file in the output folder
        with open(os.path.join(output_folder, filename), "wb") as f:
            pickle.dump(combined_data, f)
