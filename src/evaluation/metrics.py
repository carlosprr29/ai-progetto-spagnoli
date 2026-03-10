# src/evaluation/metrics.py

from sklearn.metrics import accuracy_score as _acc, precision_recall_fscore_support, confusion_matrix as _cm

def accuracy_score(labels, preds):
    """Calcula la exactitud (accuracy)."""
    return float(_acc(labels, preds))

def precision_recall_f1(labels, preds):
    """Calcula precisión, recall y F1 (binary)."""
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return float(p), float(r), float(f1)

def confusion_matrix(labels, preds):
    """Devuelve la matriz de confusión 2x2."""
    return _cm(labels, preds)

def classification_report(labels, preds):
    """Genera reporte de clasificación (texto) por clases."""
    p, r, f1, support = precision_recall_fscore_support(labels, preds, average=None)
    report = "Clases: Fake (1), Real (0)\n"
    report += f"Precision  Fake: {p[1]:.3f}, Real: {p[0]:.3f}\n"
    report += f"Recall     Fake: {r[1]:.3f}, Real: {r[0]:.3f}\n"
    report += f"F1-score   Fake: {f1[1]:.3f}, Real: {f1[0]:.3f}\n"
    report += f"Apoyo      Fake: {support[1]}, Real: {support[0]}\n"
    return report
