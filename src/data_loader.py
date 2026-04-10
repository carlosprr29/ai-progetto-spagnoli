import pandas as pd
import os
import re
from datasets import load_dataset

def clean_text_comprehensive(text, is_title=False):
    if not isinstance(text, str) or text.strip() == "": 
        return "no content"

    # A. Datelines & Headers: SOLO para el cuerpo (text), NO para títulos
    # Así evitamos cargaros títulos como "Pence: My story"
    if not is_title:
        text = re.sub(r'^[^-:]*[-:]\s*', '', text)

    # B. Web Noise (URLs y menciones) - Esto sí va en ambos
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\S+', '', text)

    # C. Agency De-biasing (Neutralizar marcas) - Esto sí va en ambos
    marcas = r'\b(Reuters|Breitbart|InfoWars|CNN|Fox News|BBC|Associated Press|AP|RT|Sputnik)\b'
    text = re.sub(marcas, 'the news outlet', text, flags=re.IGNORECASE)

    # D. End of text signatures (Solo suele estar en el texto)
    if not is_title:
        text = re.sub(r'(?i)read more|source\s*[:\-].*', '', text)

    # E. Normalization: Minúsculas solo para títulos (para evitar el sesgo de GRITAR)
    if is_title:
        text = text.lower()

    return text.strip() if text.strip() != "" else "no content"

def load_and_clean_welfake():
    """Fetches, cleans, and tags WELFake from Hugging Face."""
    print(" Loading WELFake from Hugging Face...")
    hf_dataset = load_dataset("davanstrien/WELFake", split='train')
    df = hf_dataset.to_pandas()

    # Clean nulls and duplicates
    df = df.dropna(subset=['text', 'label']).drop_duplicates()
    
    print("   ↳ Applying comprehensive cleaning to WELFake...")
    df['text'] = df['text'].apply(lambda x: clean_text_comprehensive(x, is_title=False))
    df['title'] = df['title'].fillna('no title').apply(lambda x: clean_text_comprehensive(x, is_title=True))

    df_sub = df[['title', 'text', 'label']].copy()
    df_sub['source'] = 'WELFake'
    return df_sub

def load_and_clean_isot(base_path):
    """Loads, cleans bias, and tags ISOT from local CSVs."""
    path_true = os.path.join(base_path, 'data', 'True.csv')
    path_fake = os.path.join(base_path, 'data', 'Fake.csv')
    
    if not (os.path.exists(path_true) and os.path.exists(path_fake)):
        print("❌ Error: ISOT files not found.")
        return None

    df_true = pd.read_csv(path_true)
    df_fake = pd.read_csv(path_fake)
    
    # Standardizing labels: 0 = REAL, 1 = FAKE
    df_true['label'] = 0
    df_fake['label'] = 1
    df_isot = pd.concat([df_true, df_fake])

    print("   ↳ Applying comprehensive cleaning to ISOT...")
    df_isot['text'] = df_isot['text'].apply(lambda x: clean_text_comprehensive(x, is_title=False))
    df_isot['title'] = df_isot['title'].fillna('no title').apply(lambda x: clean_text_comprehensive(x, is_title=True))
    
    df_sub = df_isot[['title', 'text', 'label']].copy()
    df_sub['source'] = 'ISOT'
    return df_sub

def create_fusion_dataset(base_path):
    """Orchestrates the full process and creates the 'total' column."""
    df_welfake = load_and_clean_welfake()
    df_isot = load_and_clean_isot(base_path)
    
    if df_welfake is None or df_isot is None: return None

    print(" Merging both dataframes...")
    df_fusion = pd.concat([df_welfake, df_isot]).reset_index(drop=True)

    # Create 'total' column for Baseline and BERT models
    print(" Creating 'total' column (Title + Text)...")
    df_fusion['total'] = df_fusion['title'] + " " + df_fusion['text']

    # Save results
    full_output_path = os.path.join(base_path, 'data', 'final_fused_dataset.csv')
    df_fusion.to_csv(full_output_path, index=False)
    
    print(f"🚀 FUSION COMPLETE: {len(df_fusion)} total articles.")
    return df_fusion