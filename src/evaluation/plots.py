# src/evaluation/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def plot_confusion_matrix(cm, classes):
    """
    Grafica la matriz de confusión dada.
    """
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

def plot_precision_recall_curve(labels, scores):
    """
    Grafica la curva Precision-Recall para la clase positiva.
    """
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.plot(recall, precision, marker='.')
    plt.title("Curva Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def plot_roc_curve(labels, scores):
    """
    Grafica la curva ROC y calcula el AUC.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, marker='.')
    plt.title(f"Curva ROC (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def plot_length_distribution(lengths):
    """
    Histograma de longitudes de texto.
    """
    plt.hist(lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribución de longitudes de noticias")
    plt.xlabel("Longitud (número de caracteres)")
    plt.ylabel("Frecuencia")
    plt.show()
