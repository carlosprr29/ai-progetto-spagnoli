# src/utils.py

import os
import json
import random
import numpy as np
import torch
import pandas as pd
import shutil
import logging
import re

def set_seed(seed: int) -> None:
    """Fija la semilla aleatoria en Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Devuelve 'cuda' si hay GPU disponible, sino 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_read_csv(path: str, encoding: str = 'utf-8', chunksize=None) -> pd.DataFrame:
    """
    Lee CSV con manejo de encoding y opcional chunks.
    """
    try:
        if chunksize:
            return pd.read_csv(path, encoding=encoding, chunksize=chunksize)
        else:
            return pd.read_csv(path, encoding=encoding)
    except Exception as e:
        # Intentar con otro encoding
        return pd.read_csv(path, encoding='latin-1')

def ensure_dir(path: str) -> None:
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)

def save_json(data: dict, filename: str) -> None:
    """Guarda un diccionario en un archivo JSON."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(filename: str) -> dict:
    """Carga un archivo JSON y devuelve su contenido."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def copy_to_drive(src_path: str, dest_path: str) -> None:
    """
    Copia archivos/carpetas a Google Drive (Colab). Usa shutil forzosamente.
    """
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
    else:
        shutil.copy(src_path, dest_path)

def path_join(*parts) -> str:
    """Une partes de ruta de forma portable."""
    return os.path.join(*parts)

def concat_title_text(row: pd.Series) -> str:
    """
    Concatena columnas 'title' y 'text' (maneja valores nulos).
    """
    title = row.get("title", "")
    text  = row.get("text", "")
    if pd.isna(title): title = ""
    if pd.isna(text): text = ""
    return f"{title} {text}".strip()

def safe_fillna(series: pd.Series, value: str) -> pd.Series:
    """
    Rellena NaN con un valor por defecto en un pandas Series.
    """
    return series.fillna(value)

def remove_reuters_signature(text: str) -> str:
    """
    Elimina firmas de Reuters al inicio (ej. 'LONDON (Reuters) - ').
    """
    return re.sub(r'^[^(]*(\(Reuters\)\s*[-:]\s*)', '', str(text))

def setup_logger(name: str) -> logging.Logger:
    """
    Configura un logger con nivel INFO y formato sencillo.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


def test_set_seed():
    import torch
    from src.utils import set_seed
    set_seed(0)
    a = np.random.rand()
    set_seed(0)
    b = np.random.rand()
    assert a == b

def test_get_device():
    from src.utils import get_device
    dev = get_device()
    assert str(dev) in ["cpu", "cuda"]

def test_safe_read_csv(tmp_path):
    import pandas as pd
    from src.utils import safe_read_csv
    df = pd.DataFrame({"a":[1,2]})
    file = tmp_path/"fichero.csv"
    df.to_csv(file, index=False)
    df2 = safe_read_csv(str(file))
    assert df2.equals(df)

def test_json(tmp_path):
    from src.utils import save_json, load_json
    data = {"x":123}
    file = tmp_path/"data.json"
    save_json(data, str(file))
    assert load_json(str(file)) == data

def test_copy_to_drive(tmp_path):
    import os
    from src.utils import copy_to_drive
    # Crear carpeta con archivo
    src_dir = tmp_path/"srcd"
    src_dir.mkdir()
    f = src_dir/"a.txt"
    f.write_text("hola")
    dest_dir = tmp_path/"dest"
    copy_to_drive(str(src_dir), str(dest_dir))
    assert os.path.exists(str(dest_dir/"a.txt"))
