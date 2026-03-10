# src/models/utils.py

import random, numpy as np, torch

def set_seed(seed: int) -> None:
    """Fija la semilla para Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Devuelve 'cuda' si hay GPU disponible, sino 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(model_name: str):
    """Carga un tokenizador HuggingFace dado el nombre del modelo."""
    from transformers import BertTokenizer  # o cargar según model_name
    return BertTokenizer.from_pretrained(model_name)

def compute_metrics(pred):
    """Callback de HF Trainer: retorna accuracy, precision, recall, f1."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

from src.models.utils import set_seed
set_seed(42)
