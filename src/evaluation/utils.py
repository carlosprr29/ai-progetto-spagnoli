# src/evaluation/utils.py

import pandas as pd

def load_results(csv_path):
    """
    Carga etiquetas y predicciones desde un CSV con columnas 'label' y 'pred'.
    """
    df = pd.read_csv(csv_path)
    labels = df['label'].astype(int).tolist()
    preds  = df['pred' ].astype(int).tolist()
    return labels, preds
