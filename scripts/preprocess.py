from azure.storage.blob import BlobServiceClient
import os
import pandas as pd
from PIL import Image
from io import BytesIO

connection_string = ""# Replace with your connection string
container_name = "" # Replace with your container name

# Directory to save downloaded images
download_dir = "./data"

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Ensure the download directory exists
os.makedirs(download_dir, exist_ok=True)

# Download metadata.csv
blob_client = container_client.get_blob_client("metadata.csv")
with open(os.path.join(download_dir, "metadata.csv"), "wb") as f:
    f.write(blob_client.download_blob().readall())

# Load metadata
metadata_df = pd.read_csv(os.path.join(download_dir, "metadata.csv"))
print("Metadata loaded:", metadata_df.head())

# Mapping of grades to folder names
folder_mapping = {
    2: "grade-2",
    5: "grade-5",
    10: "grade-10"
}

def download_images():
    for _, row in metadata_df.iterrows():
        image_name = row["image_name"]
        grade = row["grade"]
        folder = folder_mapping.get(grade)
        blob_path = f"{folder}/{image_name}"
        blob_client = container_client.get_blob_client(blob_path)
        
        img_data = blob_client.download_blob().readall()
        img = Image.open(BytesIO(img_data))
        img.save(os.path.join(download_dir, image_name))
        print(f"Downloaded: {image_name}")

if __name__ == "__main__":
    download_images()