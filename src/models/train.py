# src/models/train.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.models.utils import set_seed, compute_metrics, get_device
import src.models.config as config

def train_model(texts_train, texts_test, labels_train, labels_test):
    """
    Entrena un modelo BERT para clasificación de noticias.
    Args:
        texts_train, texts_test: Listas de strings (título + texto completo).
        labels_train, labels_test: Listas de int (0=real, 1=fake).
    Returns:
        (accuracy, modelo_entrenado, tokenizador).
    """
    # Fijar semilla y dispositivo
    set_seed(42)
    device = get_device()
    
    # Cargar tokenizador y modelo preentrenado
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=2
    )
    model.to(device)
    
    # Tokenizar los textos (padding & truncation)
    enc_train = tokenizer(texts_train, padding=True, truncation=True, max_length=config.MAX_LENGTH)
    enc_test  = tokenizer(texts_test, padding=True, truncation=True, max_length=config.MAX_LENGTH)
    
    # Crear Dataset Torch compatible
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]).to(device) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx]).to(device)
            return item
        def __len__(self):
            return len(self.labels)
    
    train_dataset = NewsDataset(enc_train, labels_train)
    test_dataset  = NewsDataset(enc_test, labels_test)
    
    # Configuración de entrenamiento
    args = TrainingArguments(
        output_dir=config.TRAIN_OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="no",
        fp16=True,
        logging_steps=100,
        load_best_model_at_end=False,
    )
    
    # Entrenador HuggingFace
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Entrenar
    trainer.train()
    
    # Evaluar final
    results = trainer.evaluate()
    acc = results.get("eval_accuracy", 0.0)
    print(f"Precisión final en test: {acc:.4f}")
    return acc, model, tokenizer

from src.models.train import train_model
acc, modelo, tokenizador = train_model(
    train_df["text"].tolist(),
    test_df["text"].tolist(),
    train_df["label"].tolist(),
    test_df["label"].tolist()
)
print(f"Precisión final: {acc:.4f}")
modelo.save_pretrained("models/bert_trained")
tokenizador.save_pretrained("models/bert_trained")
