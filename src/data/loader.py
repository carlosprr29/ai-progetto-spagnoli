# loader.py
import pandas as pd

def load_isot(true_path, fake_path):
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    # Homogeneizar nombres de columnas, p.ej. 'Title' vs 'title'
    df_true.rename(columns=lambda x: x.strip().lower(), inplace=True)
    df_fake.rename(columns=lambda x: x.strip().lower(), inplace=True)
    # Asignar etiquetas
    df_true['label'] = 0
    df_fake['label'] = 1
    # Unir y resetear índices
    df = pd.concat([df_true, df_fake], ignore_index=True)
    return df[['title','text','label']]  # Sólo columnas relevantes

def load_welfake(path):
    df = pd.read_csv(path)
    df.rename(columns={'Title':'title','Text':'text','Label':'label'}, inplace=True)
    # Algunas versiones incluyen un índice llamado 'Serial'; ignorar si existe
    cols = [c.lower() for c in df.columns]
    for col in ['serial','none']:
        if col in cols: 
            df.drop(columns=col, inplace=True, errors='ignore')
    return df[['title','text','label']]
