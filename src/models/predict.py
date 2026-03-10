# src/models/predict.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

def predict_news(text, tokenizer, model):
    """
    Predice la etiqueta de fake news (0=real, 1=falso) para un texto dado.
    Args:
        text: String de la noticia (título + texto).
        tokenizer: Tokenizador ya cargado (p.ej. BertTokenizer).
        model: Modelo BERT cargado (PreTrainedModel).
    Returns:
        (label, prob): label=0 o 1, prob=confianza de la predicción.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(model.device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=1).cpu().numpy()[0]
    label = int(probs.argmax())
    prob = float(probs[label])
    return label, prob


from src.models.predict import predict_news
from src.models.train import train_model

# Suponiendo que ya entrenamos o cargamos un modelo
_, modelo, tokenizador = train_model(textos_train, textos_test, labels_train, labels_test)
texto_nuevo = "Nuevo estudio confirma baja incidencia de X en pacientes..."
label, prob = predict_news(texto_nuevo, tokenizador, modelo)
print(f"Predicción: {label} (prob {prob:.2%})")

