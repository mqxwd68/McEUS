"""
Utility script to automatically download and extract pre-trained weights from Zenodo.
Provides a permanent, quota-free download channel for the McEUS pipeline.
"""

import os
import zipfile
import urllib.request
import sys

# ==========================================
# Configuration: Replace with your actual Zenodo Record ID
# ==========================================
RECORD_ID = "20193485"
FILE_NAME = "weights.zip"
TARGET_DIR = "weights"

# Construct the official Zenodo direct download API URL
DOWNLOAD_URL = f"https://zenodo.org/records/{RECORD_ID}/files/{FILE_NAME}?download=1"


def download_with_progress():
    """Download the file using urllib with a tqdm progress bar."""
    print(f"📥 Downloading pre-trained weights from Zenodo (Record ID: {RECORD_ID})...")

    try:
        from tqdm import tqdm
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=FILE_NAME) as t:
            urllib.request.urlretrieve(DOWNLOAD_URL, filename=FILE_NAME, reporthook=t.update_to)

    except ImportError:
        # Fallback if tqdm is somehow not installed
        print("⏳ Downloading... (Please wait, this may take a while as tqdm is not found)")
        urllib.request.urlretrieve(DOWNLOAD_URL, filename=FILE_NAME)

    print("\n✅ Download successfully completed.")


def extract_weights():
    """Extract the downloaded zip archive."""
    print(f"📦 Extracting {FILE_NAME}...")
    try:
        with zipfile.ZipFile(FILE_NAME, 'r') as zip_ref:
            zip_ref.extractall(".")  # Extract to current root directory
        print("✅ Extraction complete.")
    except zipfile.BadZipFile:
        print("❌ Error: The downloaded file is corrupted or not a valid zip archive.")
        sys.exit(1)


def main():
    # 1. Check if the target directory already exists
    if os.path.exists(TARGET_DIR) and os.path.isdir(TARGET_DIR):
        print(f"✅ Directory '{TARGET_DIR}' already exists. Setup is already complete.")
        return

    # 2. Download the zip file if it doesn't exist locally
    if not os.path.exists(FILE_NAME):
        download_with_progress()
    else:
        print(f"✅ '{FILE_NAME}' already exists locally. Skipping download.")

    # 3. Extract the archive
    if os.path.exists(FILE_NAME):
        extract_weights()

        # Optionally clean up the zip file to save disk space
        os.remove(FILE_NAME)
        print(f"🗑️ Removed temporary archive '{FILE_NAME}'.")

    print("\n🎉 All weights are successfully prepared and ready for inference!")


if __name__ == "__main__":
    main()