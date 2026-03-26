import pandas as pd
import os
import re
from datasets import load_dataset

def remove_reuters_bias(text):
    """
    Removes patterns like 'WASHINGTON (Reuters) -' at the start of ISOT articles.
    """
    # Removes patterns before the first dash if it contains 'Reuters' or similar
    pattern = r'^.*?\(Reuters\)\s*-\s*|^.*?\s*-\s*'
    return re.sub(pattern, '', str(text), count=1, flags=re.IGNORECASE)

def load_and_clean_welfake():
    """
    Fetches, cleans, and tags WELFake from Hugging Face.
    """
    print(" Loading WELFake from Hugging Face...")
    hf_dataset = load_dataset("search67/WELFake", split='train')
    df = hf_dataset.to_pandas()
    
    # 1. Clean nulls and duplicates
    df = df.dropna(subset=['text', 'label']).drop_duplicates()
    
    # 2. Select columns and add source tag
    df_sub = df[['text', 'label']].copy()
    df_sub['source'] = 'WELFake'
    
    print(f"✅ WELFake loaded and tagged: {len(df_sub)} rows.")
    return df_sub

def load_and_clean_isot(base_path):
    """
    Loads, cleans bias, and tags ISOT from local CSVs.
    """
    path_true = os.path.join(base_path, 'data', 'True.csv')
    path_fake = os.path.join(base_path, 'data', 'Fake.csv')
    
    if not (os.path.exists(path_true) and os.path.exists(path_fake)):
        print("❌ Error: ISOT files not found.")
        return None

    df_true = pd.read_csv(path_true)
    df_fake = pd.read_csv(path_fake)
    df_true['label'] = 1
    df_fake['label'] = 0
    df_isot = pd.concat([df_true, df_fake])

    # 1. Apply Reuters bias removal
    print(" Cleaning ISOT source bias...")
    df_isot['text'] = df_isot['text'].apply(remove_reuters_bias)
    
    # 2. Select columns and add source tag
    df_sub = df_isot[['text', 'label']].copy()
    df_sub['source'] = 'ISOT'
    
    print(f"✅ ISOT loaded and cleaned: {len(df_sub)} rows.")
    return df_sub

def create_fusion_dataset(base_path):
    """
    Orchestrates the full process: Load -> Clean -> Merge -> Save Full & Samples.
    """
    # 1. Get cleaned data
    df_welfake = load_and_clean_welfake()
    df_isot = load_and_clean_isot(base_path)
    
    if df_welfake is None or df_isot is None:
        return None

    # 2. Merging
    print(" Merging both dataframes...")
    df_fusion = pd.concat([df_welfake, df_isot]).reset_index(drop=True)

    # 3. SAVE FULL DATASET
    full_output_path = os.path.join(base_path, 'data', 'final_fused_dataset.csv')
    df_fusion.to_csv(full_output_path, index=False)
    print(f" Full dataset saved to: {full_output_path}")

    # 4. SAVE SAMPLES CHECK (New!)
    # We take a small sample (e.g., 50 rows) to quickly inspect results
    sample_df = df_fusion.sample(min(100, len(df_fusion)), random_state=42)
    sample_path = os.path.join(base_path, 'data', 'fusion_samples_check.csv')
    sample_df.to_csv(sample_path, index=False)
    print(f" Check samples saved to: {sample_path}")
    
    print(f"🚀 FUSION COMPLETE: {len(df_fusion)} total articles.")
    return df_fusion
