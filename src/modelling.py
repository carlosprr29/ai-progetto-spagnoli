import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# --- 1. BASELINE MODEL (Classic Machine Learning) ---

def train_baseline_expert(df, name, text_col='total'):
    """
    Trains a Logistic Regression specialist as a Baseline comparison.
    """
    print(f"\n--- Training Baseline Specialist: {name} ---")

    # Split 80/20 with stratification to maintain class balance
    train_data, test_data = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['label']
    )

    # Simple but effective TF-IDF + Logistic Regression pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    model.fit(train_data[text_col], train_data['label'])

    preds = model.predict(test_data[text_col])
    acc = accuracy_score(test_data['label'], preds)
    print(f"✅ Baseline Expert '{name}' ready. Evaluation Accuracy: {acc:.4f}")
    
    return model


# --- 2. ADVANCED MODEL (Deep Learning / BERT) ---

def train_bert_expert(train_text, test_text, train_y, test_y, name, max_len=256):
    """
    Fine-tunes a BERT model for sequence classification.
    """
    print(f"\n--- Training BERT Expert: {name} (Max Sequence Length: {max_len}) ---")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 1. Tokenization (Optimized by converting to list once)
    print("   ↳ Tokenizing datasets...")
    train_encodings = tokenizer(train_text.tolist(), truncation=True, padding=True, max_length=max_len)
    test_encodings = tokenizer(test_text.tolist(), truncation=True, padding=True, max_length=max_len)

    # 2. PyTorch Dataset Class
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

    # 3. Load Model and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    # 4. Hyperparameters 
    # Added fp16 and gradient accumulation for better performance on Colab Free Tier
    training_args = TrainingArguments(
        output_dir=f"./results_{name}",
        num_train_epochs=2,
        per_device_train_batch_size=8,   # Smaller batch to prevent Out-of-Memory (OOM)
        gradient_accumulation_steps=2,    # Effective batch size remains 16
        learning_rate=2e-5,
        fp16=True,                        # Mixed precision for faster training
        eval_strategy="epoch",            # Updated from 'evaluation_strategy'
        save_strategy="no",               # Don't save checkpoints to save space
        logging_steps=50,
        weight_decay=0.01                 # Regularization to prevent overfitting
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=NewsDataset(train_encodings, train_y.tolist()), 
        eval_dataset=NewsDataset(test_encodings, test_y.tolist()),
        compute_metrics=compute_metrics
    )

    print(f"   ↳ Starting fine-tuning for {name}...")
    trainer.train()
    
    print(f"✅ BERT Expert '{name}' training complete.")
    return trainer, model, tokenizer