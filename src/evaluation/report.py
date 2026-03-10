# src/evaluation/report.py

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def generate_report(y_true, y_pred):
    """
    Genera un informe completo de evaluación del modelo.

    Parámetros
    ----------
    y_true : list o array
        Etiquetas reales del dataset de test.

    y_pred : list o array
        Etiquetas predichas por el modelo.

    Returns
    -------
    report_text : str
        Texto con accuracy y classification report.
    """

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Classification report
    report = classification_report(y_true, y_pred)

    # Construimos el texto final
    report_text = ""
    report_text += "==============================\n"
    report_text += "MODEL EVALUATION REPORT\n"
    report_text += "==============================\n\n"

    report_text += f"Accuracy: {acc:.4f}\n\n"

    report_text += "Classification Report:\n"
    report_text += report
    report_text += "\n"

    return report_text


def print_report(y_true, y_pred):
    """
    Imprime el informe en consola.
    """

    report_text = generate_report(y_true, y_pred)

    print(report_text)


def save_report(y_true, y_pred, filepath):
    """
    Guarda el informe en un archivo de texto.

    Parámetros
    ----------
    filepath : str
        Ruta donde guardar el informe (ej: results/report.txt)
    """

    report_text = generate_report(y_true, y_pred)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report saved in: {filepath}")
