import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# --- 1. BASELINE MODEL (Machine Learning Clásico) ---

def train_baseline_expert(df, name, text_col='total'):
    """
    Trains a Logistic Regression specialist (Baseline).
    """
    print(f"\n 🧠 Training Baseline Specialist: {name}...")

    # Split 80/20
    train_data, test_data = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['label']
    )

    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    model.fit(train_data[text_col], train_data['label'])

    preds = model.predict(test_data[text_col])
    acc = accuracy_score(test_data['label'], preds)
    print(f"✅ {name} Baseline Expert ready. Accuracy: {acc:.4f}")
    
    return model


# --- 2. ADVANCED MODEL (Deep Learning / BERT) ---

def train_bert_expert(train_text, test_text, train_y, test_y, name, max_len=256):
    print(f"\n 🔥 Training BERT Expert: {name} (Max Length: {max_len})")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 1. Tokenización (Optimizada convirtiendo a lista una vez)
    train_encodings = tokenizer(train_text.tolist(), truncation=True, padding=True, max_length=max_len)
    test_encodings = tokenizer(test_text.tolist(), truncation=True, padding=True, max_length=max_len)

    # 2. Dataset Clase (Igual que la tuya, pero dentro para evitar conflictos)
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    # 3. Cargar modelo y mover a GPU
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. Hiperparámetros (Añadido fp16 y gradient_accumulation para ir más rápido)
    training_args = TrainingArguments(
        output_dir=f"./results_{name}",
        num_train_epochs=2,
        per_device_train_batch_size=8, # Bajamos a 8 para evitar errores de memoria en Colab
        gradient_accumulation_steps=2, # Pero acumulamos 2 pasos (total 16 efectivos)
        learning_rate=2e-5,
        fp16=True, 
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=NewsDataset(train_encodings, train_y.tolist()), 
        eval_dataset=NewsDataset(test_encodings, test_y.tolist()),
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer, model, tokenizer