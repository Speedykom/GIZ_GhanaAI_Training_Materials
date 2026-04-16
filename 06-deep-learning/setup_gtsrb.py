#!/usr/bin/env python3
"""
Setup script to download and prepare GTSRB dataset for DL2 notebook.
Run this before rendering the notebook.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import shutil
import numpy as np
from glob import glob


def download_and_prepare_gtsrb():
    """Download and prepare GTSRB dataset."""

    # Check if already downloaded
    if Path("data").exists():
        print("✅ Data directory already exists. Skipping download.")
        return

    # Download dataset
    url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    zip_path = "GTSRB_Final_Training_Images.zip"

    if not Path(zip_path).exists():
        print("📥 Downloading GTSRB dataset (~300MB)...")
        print("   This may take a few minutes...")
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete!")

    # Extract
    if not Path("GTSRB").exists():
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ Extraction complete!")

    # Clean up zip
    if Path(zip_path).exists():
        os.remove(zip_path)
        print("🧹 Cleaned up zip file")

    # Define class mapping (class_id -> class_name)
    # From GTSRB dataset documentation
    class_mapping = {12: "priority_road", 13: "give_way", 14: "stop", 17: "no_entry"}

    # Create data directory structure
    DATA_DIR = Path("data")
    DATASETS = ["train", "val", "test"]

    print("📁 Creating directory structure...")
    for ds in DATASETS:
        for cls in class_mapping.values():
            (DATA_DIR / ds / cls).mkdir(parents=True, exist_ok=True)

    # Get training folders
    train_folders = sorted(glob("GTSRB/Final_Training/Images/*"))

    # Copy images to organized structure
    print("📋 Organizing images into train/val/test splits...")
    for class_id, class_name in class_mapping.items():
        # Find folder for this class
        class_folder = None
        for folder in train_folders:
            if folder.endswith(f"{class_id:05d}"):
                class_folder = folder
                break

        if not class_folder:
            print(f"⚠️  Warning: Could not find folder for class {class_id}")
            continue

        # Get all images for this class
        image_paths = np.array(glob(f"{class_folder}/*.ppm"))
        print(f"   {class_name}: {len(image_paths)} images")

        # Shuffle for random split
        np.random.seed(42)
        np.random.shuffle(image_paths)

        # Split: 80% train, 10% val, 10% test
        n_train = int(0.8 * len(image_paths))
        n_val = int(0.9 * len(image_paths))

        splits = {
            "train": image_paths[:n_train],
            "val": image_paths[n_train:n_val],
            "test": image_paths[n_val:],
        }

        # Copy images
        for split_name, images in splits.items():
            for img_path in images:
                shutil.copy(img_path, f"{DATA_DIR}/{split_name}/{class_name}/")

    # Print summary
    print("\n📊 Dataset Summary:")
    for ds in DATASETS:
        count = sum(
            len(list((DATA_DIR / ds / cls).glob("*.ppm")))
            for cls in class_mapping.values()
        )
        print(f"   {ds}: {count} images")

    print("\n✅ Dataset preparation complete!")
    print("   You can now render the notebook.")


if __name__ == "__main__":
    download_and_prepare_gtsrb()
