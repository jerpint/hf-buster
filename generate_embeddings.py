import os
import zipfile
import requests
import pandas as pd
import time

from buster.documents_manager import DeepLakeDocumentsManager

from buster.docparser import get_all_documents
from buster.parser import HuggingfaceParser

hf_transformers_zip_url = "https://huggingface.co/datasets/hf-doc-build/doc-build/resolve/main/transformers/main.zip"


def download_and_unzip(zip_url, target_dir, overwrite=False):
    """Download a zip file from zip_url and unzip it to target_dir.

    # Example usage
    zip_url = "https://example.com/example.zip"
    target_dir = "downloaded_files"
    download_and_unzip(zip_url, target_dir, overwrite=True)

    ChatGPT generated.
    """
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get the filename from the zip_url
    zip_filename = os.path.basename(zip_url)
    target_path = os.path.join(target_dir, zip_filename)

    # Check if the file already exists
    if os.path.exists(target_path) and not overwrite:
        print(f"{zip_filename} already exists in the target directory.")
        return

    # Download the zip file
    response = requests.get(zip_url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"{zip_filename} downloaded successfully.")

        # Unzip the file
        with zipfile.ZipFile(target_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"{zip_filename} extracted successfully.")
    else:
        print(f"Failed to download {zip_filename}. Status code: {response.status_code}")


# Download the tranformers html pages and unzip it
download_and_unzip(zip_url=hf_transformers_zip_url, target_dir=".")

# Extract all documents from the html into a dataframe
df = get_all_documents(
    root_dir="transformers/main/en/",
    base_url="https://huggingface.co/docs/transformers/main/en/",
    parser_cls=HuggingfaceParser,
    min_section_length=100,
    max_section_length=1000,
)

# Add the source column
df["source"] = "hf_transformers"

# Save the .csv with chunks to disk
df.to_csv("hf_transformers.csv")

# Initialize the vector store
dm = DeepLakeDocumentsManager(
    vector_store_path="deeplake_store",
    overwrite=True,
    required_columns=["url", "content", "source", "title"],
)

# Add all embeddings to the vector store
dm.batch_add(
    df=df,
    batch_size=3000,
    min_time_interval=60,
    num_workers=32,
    csv_filename="embeddings.csv",
    csv_overwrite=False,
)
