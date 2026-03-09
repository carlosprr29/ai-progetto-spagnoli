"Particionamiento y muestreo"
def stratified_split(df: pd.DataFrame, label_col: str, test_size: float, seed: int):
    """
    Divide el DataFrame en train/test manteniendo proporción de `label_col` (muestreo estratificado).
    Devuelve (df_train, df_test).
    """
def sample_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Toma una muestra aleatoria de tamaño `n` preservando proporción de clases (usando stratify).
    """
from src.data.split import stratified_split

train_df, test_df = stratified_split(df, label_col='label', test_size=0.2, seed=42)
