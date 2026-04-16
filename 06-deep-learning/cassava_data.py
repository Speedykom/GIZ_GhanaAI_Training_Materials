import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import time

# Detect environment
try:
    from google.colab import files
    IN_COLAB = True
    print("Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running in local Jupyter")

# Load environment variables
if IN_COLAB:
    # Upload .env file in Colab
    if not os.path.exists('.env'):
        print("\nPlease upload your .env file:")
        uploaded = files.upload()
        if '.env' in uploaded:
            print("✓ .env file uploaded successfully")
    
    from dotenv import load_dotenv
    load_dotenv()
else:
    # Local Jupyter - load from existing .env file
    from dotenv import load_dotenv
    load_dotenv()

# Get credentials
kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

if not kaggle_username or not kaggle_key:
    print("Error: KAGGLE_USERNAME and KAGGLE_KEY not found in .env file")
else:
    # Setup credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    credentials = {"username": kaggle_username, "key": kaggle_key}
    with open(kaggle_json_path, 'w') as f:
        json.dump(credentials, f)
    os.chmod(kaggle_json_path, 0o600)
    print(f"✓ Kaggle credentials saved to {kaggle_json_path}")
    
    # Use official Kaggle API with retry logic
    api = KaggleApi()
    api.authenticate()
    
    # Dataset name
    dataset_name = 'nirmalsankalana/cassava-leaf-disease-classification'
    download_path = './cassava_dataset'
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"\n{'Retrying download...' if retry_count > 0 else 'Starting download...'} (Attempt {retry_count + 1}/{max_retries})")
            
            api.dataset_download_files(
                dataset_name,
                path=download_path,
                unzip=True,
                quiet=False
            )
            
            print("✓ Dataset downloaded successfully!")
            print(f"✓ Path to dataset files: {os.path.abspath(download_path)}")
            
            # List contents
            print("\nDataset contents:")
            for item in os.listdir(download_path):
                print(f"  - {item}")
            
            break
            
        except Exception as e:
            retry_count += 1
            print(f"✗ Download failed: {e}")
            
            if retry_count < max_retries:
                wait_time = 10 * retry_count
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("\n✗ Max retries reached. Download failed.")
                print("\nTroubleshooting tips:")
                print("1. Check your internet connection stability")
                print("2. Try downloading during off-peak hours")
                print("3. Consider manual download from Kaggle website")
                print(f"4. Dataset URL: https://www.kaggle.com/datasets/{dataset_name}")