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
# src/evaluation/utils.py (continuación)

def save_figure(fig, filename):
    """
    Guarda la figura actual de Matplotlib en 'filename'.
    """
    fig.savefig(filename, bbox_inches='tight')

def format_table(df):
    """
    Da formato a un DataFrame (alineación, decimales) para mostrar en reporte.
    """
    return df.to_string(index=False, float_format="{:.3f}".format)
