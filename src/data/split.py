import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, label_col: str = 'label',
                     test_size: float = 0.2, seed: int = 42):
    """
    Divide el DataFrame en train/test manteniendo la proporción
    de clases en `label_col` (muestreo estratificado).

    Args:
        df: DataFrame completo.
        label_col: Columna que contiene las etiquetas.
        test_size: Proporción del conjunto de test.
        seed: Semilla para reproducibilidad.

    Returns:
        Tuple (df_train, df_test).
    """

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed
    )

    return train_df, test_df


def sample_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Toma una muestra aleatoria del DataFrame.

    Args:
        df: DataFrame original.
        n: Número de muestras deseadas.
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame con una muestra aleatoria.
    """

    if n < len(df):
        return df.sample(n=n, random_state=seed)
    else:
        return df.copy()

from src.data.split import stratified_split

train_df, test_df = stratified_split(df, label_col='label', test_size=0.2, seed=42)
