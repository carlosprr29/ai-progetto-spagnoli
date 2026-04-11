import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from transformers import BertTokenizer, BertForSequenceClassification

def load_committee_models():
    """Loads all fine-tuned BERT models and returns a dictionary of experts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mapping models to compare "Title Experts" vs "Full Experts"
    paths = {
        "WELFake (Titles)": "/content/drive/MyDrive/Project_IA/model_WELFake_titles",
        "WELFake (Full)":   "/content/drive/MyDrive/Project_IA/model_WELFake_full",
        "ISOT (Titles)":    "/content/drive/MyDrive/Project_IA/model_ISOT_titles",
        "ISOT (Full)":      "/content/drive/MyDrive/Project_IA/model_ISOT_full",
        "Fusion (Titles)":  "/content/drive/MyDrive/Project_IA/model_fusion_titles",
        "Fusion (Full)":    "/content/drive/MyDrive/Project_IA/model_fusion_full"
    }

    print("🚀 Loading BERT Committee of Experts... This may take a few minutes.")
    
    # Load one common tokenizer (using Fusion Full as reference)
    try:
        tokenizer = BertTokenizer.from_pretrained(paths["Fusion (Full)"])
    except:
        print("⚠️ Warning: Could not load local tokenizer, fetching bert-base-uncased from Hub.")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load all models into a dictionary
    models = {}
    for name, path in paths.items():
        print(f"   ↳ Loading {name}...")
        models[name] = BertForSequenceClassification.from_pretrained(path).to(device).eval()

    return models, tokenizer, device


def analyze_news(title, body, experts_dict, tokenizer=None, device=None):
    """
    Orchestrates the evaluation of a single news item across all experts.
    - BERT Experts: Evaluate based on their specific training specialization.
    - Baseline Models: Evaluate across all 3 variants (Ablation test).
    """
    inputs = {
        "Title Only": title,
        "Body Text Only": body,
        "Full Context": title + " [SEP] " + body if tokenizer else title + " " + body
    }

    plot_data = []
    print("\n" + "═"*85)
    print(f" 🧪 HYBRID COMMITTEE TEST: {title[:65]}...")
    print("═"*85)

    for expert_name, model in experts_dict.items():
        # --- CASE A: BERT MODELS (Specialists) ---
        if tokenizer is not None and hasattr(model, 'config'):
            # Assign the correct input mode based on the expert's name
            mode_name = "Full Context" if "Full" in expert_name else "Title Only"
            text_input = inputs[mode_name]
            
            enc = tokenizer(text_input, truncation=True, padding=True, 
                            max_length=256, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model(**enc)
                probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()[0]
                pred = np.argmax(probs)
                conf = probs[pred]
            
            label = "FAKE" if pred == 1 else "TRUE"
            plot_data.append({'Expert': expert_name, 'Mode': mode_name, 'Confidence': conf, 'Label': label})
            print(f" ↳ {expert_name:<20} | {mode_name:<15} : {label:<5} (Conf: {conf:.2%})")

        # --- CASE B: BASELINE MODELS (Generalists - Cross-testing variants) ---
        elif hasattr(model, 'predict_proba'):
            print(f"\n {expert_name} (Baseline Cross-Test):")
            for mode_name, text_input in inputs.items():
                # Remove SEP token for classic ML models
                clean_input = text_input.replace(" [SEP] ", " ")
                pred = model.predict([clean_input])[0]
                conf = model.predict_proba([clean_input])[0][pred]
                
                label = "FAKE" if pred == 1 else "TRUE"
                plot_data.append({'Expert': expert_name, 'Mode': mode_name, 'Confidence': conf, 'Label': label})
                print(f"   ↳ {mode_name:<15} : {label:<5} (Conf: {conf:.2%})")

    if plot_data:
        _generate_evaluation_plot(plot_data)
        
    return pd.DataFrame(plot_data)

def _generate_evaluation_plot(plot_data):
    """Generates a visualization showing the confidence and verdict of each expert."""
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create the bar plot
    ax = sns.barplot(data=df_plot, x='Expert', y='Confidence', hue='Mode', palette='viridis')
    
    plt.title("Committee Confidence Analysis", fontsize=14, fontweight='bold')
    plt.ylabel("Model Confidence Level")
    plt.ylim(0, 1.2) # Extra space for annotations
    plt.legend(title="Evaluation Mode", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add text labels (Confidence + Label) on top of bars
    for i, p in enumerate(ax.patches):
        if p.get_height() > 0:
            # We fetch the label from the original plot_data list
            # Note: bar patches order can be tricky with hue, this is a simplified annotation
            height = p.get_height()
            ax.annotate(f'{height:.1%}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 10), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.show()