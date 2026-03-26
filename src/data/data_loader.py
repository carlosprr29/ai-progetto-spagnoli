import pandas as pd
import os
from datasets import load_dataset

def load_datasets(base_path):
    """
    Loads WELFake from Hugging Face and ISOT datasets from Google Drive.
    
    Args:
        base_path (str): Path to the main project folder in Drive (e.g., '/content/drive/MyDrive/TFG/')
        
    Returns:
        dict: Dictionary containing the loaded DataFrames.
    """
    isot_paths = {
        'isot_true': os.path.join(base_path, 'data', 'True.csv'),
        'isot_fake': os.path.join(base_path, 'data', 'Fake.csv')
    }
    
    data = {}
    
    try:
        print("--- Loading Datasets ---")
        
        # 1. Load WELFake from Hugging Face
        print(" Fetching WELFake from Hugging Face...")
        # We load the 'train' split and convert to Pandas
        hf_dataset = load_dataset("search67/WELFake", split='train')
        data['welfake'] = hf_dataset.to_pandas()
        print(f" WELFake loaded from HF: {len(data['welfake'])} rows.")

        # 2. Load ISOT from Google Drive (True and Fake separately)
        if os.path.exists(isot_paths['isot_true']) and os.path.exists(isot_paths['isot_fake']):
            df_true = pd.read_csv(isot_paths['isot_true'])
            df_fake = pd.read_csv(isot_paths['isot_fake'])
            
            # Labeling (1 = Real, 0 = Fake)
            df_true['label'] = 1
            df_fake['label'] = 0
            
            data['isot'] = pd.concat([df_true, df_fake]).reset_index(drop=True)
            print(f" ISOT loaded from Drive and merged: {len(data['isot'])} rows.")
        else:
            print(f" Error: Missing ISOT files at {base_path}/data/")

    except Exception as e:
        print(f" Unexpected error during loading: {e}")
        
    return data

def prepare_fusion_dataset(data):
    """
    Combines WELFake and ISOT into a single master dataset for the Fusion model.
    """
    if 'welfake' in data and 'isot' in data:
        # Standardizing column names if necessary before concatenation
        fusion_df = pd.concat([data['welfake'], data['isot']], axis=0).reset_index(drop=True)
        print(f" Fusion Dataset created: {len(fusion_df)} total rows.")
        return fusion_df
    else:
        print(" Cannot create Fusion: Base datasets are missing.")
        return None
