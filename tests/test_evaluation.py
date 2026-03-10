def test_metrics_basic():
    from src.evaluation.metrics import accuracy_score, precision_recall_f1, confusion_matrix
    labels = [0,1,0,1]
    preds  = [0,1,0,0]
    assert accuracy_score(labels, preds) == 0.75
    p, r, f1 = precision_recall_f1(labels, preds)
    assert round(p, 2) == 1.00
    assert round(r, 2) == 0.50
    assert confusion_matrix(labels, preds).shape == (2,2)

def test_load_results(tmp_path):
    import pandas as pd
    from src.evaluation.utils import load_results
    data = {"label": [0,1,1], "pred": [0,0,1]}
    df = pd.DataFrame(data)
    file = tmp_path/"results.csv"
    df.to_csv(file, index=False)
    labels, preds = load_results(str(file))
    assert labels == [0,1,1] and preds == [0,0,1]
