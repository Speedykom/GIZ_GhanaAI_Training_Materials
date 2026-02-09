# Quick test to verify dataset loads correctly
import torch
from torchvision import datasets, transforms
from pathlib import Path
import time

data_dir = Path("cassava_dataset/data")

print("Testing dataset loading...")
start = time.time()

# Try loading just the dataset metadata
try:
    dataset = datasets.ImageFolder(root=data_dir)
    print(f"✓ Dataset loaded successfully")
    print(f"✓ Total images: {len(dataset):,}")
    print(f"✓ Classes: {len(dataset.classes)}")
    print(f"✓ Time: {time.time() - start:.2f}s")
    print("\nDataset is ready for students!")
except Exception as e:
    print(f"✗ Error: {e}")
