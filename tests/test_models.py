import pytest
import torch
from src.models.train import train_model
from src.models.predict import predict_news

def test_train_model_dummy():
    # Datos ficticios mínimos (2 ejemplos)
    texts_train = ["Hola mundo", "Noticias falsas"]
    labels_train = [0, 1]
    texts_test  = ["Prueba real", "Prueba fake"]
    labels_test = [0, 1]
    acc, model, tokenizer = train_model(texts_train, texts_test, labels_train, labels_test)
    assert 0.0 <= acc <= 1.0
    assert isinstance(model, object)
    assert isinstance(tokenizer, object)

def test_predict_news_consistency(tmp_path):
    # Entrenar un modelo sencillo o cargar uno fijo
    texts = ["Noticia verdadera", "Noticia inventada"]
    labels = [0, 1]
    _, model, tokenizer = train_model(texts, texts, labels, labels)
    label, prob = predict_news("Noticia inventada", tokenizer, model)
    assert label in (0,1)
    assert 0.0 <= prob <= 1.0
