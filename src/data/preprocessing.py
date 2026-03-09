# preprocessing.py

import re
import pandas as pd
from typing import List

def remove_reuters_signature(text: str) -> str:
    """
    Elimina la firma de agencia Reuters al inicio del texto,
    p.ej. 'LONDON (Reuters) - ...'.
    """
    return re.sub(r'^[A-Za-zÀ-ÿ\s]+\(Reuters\)\s*-\s*', '', text)


def clean_text(text: str) -> str:
    """
    Normaliza el texto eliminando saltos de línea, URLs,
    caracteres especiales y espacios duplicados.
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s¡¿.,;:!?\'"]', '', text)

    return text.lower().strip()


def drop_missing(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Elimina filas con valores NA en las columnas indicadas.
    """
    return df.dropna(subset=columns)


def deduplicate(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """
    Elimina filas duplicadas según las columnas indicadas.
    """
    return df.drop_duplicates(subset=subset)


from src.data.preprocessing import remove_reuters_signature, clean_text, drop_missing, deduplicate

df = load_isot("Project_IA/True.csv", "Project_IA/Fake.csv")

df['text'] = df['text'].apply(remove_reuters_signature).apply(clean_text)

df = drop_missing(df, ['title','text'])
df = deduplicate(df, ['title','text'])




