#!/usr/bin/env python3
"""
Setup script for Ghana Housing Regression Exercises
Downloads the kaggle dataset for exercises_regression.qmd

Usage:
    python setup_kaggle.py

This script will:
1. Try to download from Kaggle API (requires ~/.kaggle/kaggle.json)
2. Extract the CSV file
3. Verify the data is ready
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

def check_kaggle_credentials():
    """Check if kaggle.json exists in ~/.kaggle/"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if kaggle_json.exists():
        print("✓ Found Kaggle credentials at ~/.kaggle/kaggle.json")
        return True
    else:
        print("\n⚠️  Kaggle credentials not found!")
        print("\nTo set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/account/settings/api")
        print("  2. Click 'Create New API Token' (downloads kaggle.json)")
        print("  3. Move file to ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  5. Then rerun this script")
        return False

def download_kaggle_dataset():
    """Download dataset using kaggle CLI"""
    print("\nDownloading Ghana house rental dataset...")

    try:
        # Use kaggle CLI
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'epigos/ghana-house-rental-dataset', '-q'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✓ Download complete")
            return True
        else:
            print(f"✗ Download failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("✗ 'kaggle' command not found")
        print("  Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def extract_dataset():
    """Extract the downloaded ZIP file"""
    zip_path = Path('ghana-house-rental-dataset.zip')

    if not zip_path.exists():
        print("✗ ZIP file not found")
        return False

    try:
        print("\nExtracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('.')
        print("✓ Extraction complete")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def verify_data():
    """Verify the CSV file exists and has data"""
    csv_path = Path('house_rentals.csv')

    if not csv_path.exists():
        print("✗ house_rentals.csv not found")
        return False

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\n✓ Data verified!")
        print(f"  - Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  - Price range: GHS {df['price'].min():,} - GHS {df['price'].max():,}")
        print(f"  - Top location: {df['location'].value_counts().index[0]}")
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

def main():
    print("=" * 70)
    print("Ghana Housing Dataset Setup")
    print("=" * 70)

    # Check if data already exists
    if Path('house_rentals.csv').exists():
        print("\n✓ Data already exists!")
        if verify_data():
            print("\n✅ You're all set! Run: jupyter notebook exercises_regression.qmd")
            return 0

    # Check Kaggle credentials
    if not check_kaggle_credentials():
        return 1

    # Download
    if not download_kaggle_dataset():
        print("\n✗ Failed to download. Try manual download:")
        print("  https://www.kaggle.com/datasets/epigos/ghana-house-rental-dataset")
        return 1

    # Extract
    if not extract_dataset():
        return 1

    # Verify
    if not verify_data():
        return 1

    print("\n✅ Setup complete! Ready to run exercises_regression.qmd")
    return 0

if __name__ == '__main__':
    sys.exit(main())
