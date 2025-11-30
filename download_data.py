"""
Data Download Script for FEVER Dataset

Downloads training data, dev data, test data, and optionally the Wikipedia dump.
"""

import os
import requests
import zipfile
from tqdm import tqdm
import config


def download_file(url, destination):
    """Download a file with progress bar."""
    print(f"Downloading {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Downloaded to {destination}")


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Extracted to {extract_to}")


def download_fever_data():
    """Download all FEVER dataset files."""

    # Create data directory
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Download training data
    if not os.path.exists(config.TRAIN_FILE):
        download_file(config.TRAIN_URL, config.TRAIN_FILE)
    else:
        print(f"{config.TRAIN_FILE} already exists, skipping.")

    # Download dev data
    if not os.path.exists(config.DEV_FILE):
        download_file(config.DEV_URL, config.DEV_FILE)
    else:
        print(f"{config.DEV_FILE} already exists, skipping.")

    # Download test data
    if not os.path.exists(config.TEST_FILE):
        download_file(config.TEST_URL, config.TEST_FILE)
    else:
        print(f"{config.TEST_FILE} already exists, skipping.")

    # Download Wikipedia dump if enabled
    if config.DOWNLOAD_WIKIPEDIA:
        wiki_zip = f"{config.DATA_DIR}/wiki-pages.zip"

        if not os.path.exists(config.WIKI_DIR):
            if not os.path.exists(wiki_zip):
                print("\n" + "="*80)
                print("WARNING: Downloading Wikipedia dump (approximately 50GB)")
                print("This may take a long time depending on your internet connection.")
                print("="*80 + "\n")
                download_file(config.WIKI_URL, wiki_zip)

            extract_zip(wiki_zip, config.DATA_DIR)
        else:
            print(f"{config.WIKI_DIR} already exists, skipping Wikipedia download.")
    else:
        print("\nSkipping Wikipedia download (DOWNLOAD_WIKIPEDIA=False in config.py)")
        print("The system will assume correct evidence is provided during validation.")

    print("\n" + "="*80)
    print("Data download complete!")
    print("="*80)


if __name__ == "__main__":
    download_fever_data()
