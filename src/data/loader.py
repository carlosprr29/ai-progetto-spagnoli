import pandas as pd


def load_isot(true_path: str, fake_path: str) -> pd.DataFrame:
    """
    Carga los CSV del dataset ISOT (noticias reales y falsas) y devuelve
    un DataFrame unificado.

    Args:
        true_path: Ruta al CSV de noticias reales.
        fake_path: Ruta al CSV de noticias falsas.

    Returns:
        DataFrame con columnas ['title','text','label'],
        donde label = 0 (real) y label = 1 (fake).
    """

    # Cargar datasets
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # Homogeneizar nombres de columnas
    df_true.rename(columns=lambda x: x.strip().lower(), inplace=True)
    df_fake.rename(columns=lambda x: x.strip().lower(), inplace=True)

    # Asignar etiquetas
    df_true['label'] = 0
    df_fake['label'] = 1

    # Unir datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Mantener solo columnas relevantes
    return df[['title', 'text', 'label']]


def load_welfake(path: str) -> pd.DataFrame:
    """
    Carga el dataset WELFake y devuelve un DataFrame con columnas esenciales.

    Args:
        path: Ruta al CSV WELFake.

    Returns:
        DataFrame con columnas ['title','text','label'].
    """

    df = pd.read_csv(path)

    # Normalizar nombres de columnas
    df.rename(columns={'Title': 'title', 'Text': 'text', 'Label': 'label'}, inplace=True)

    # Algunas versiones tienen columnas innecesarias
    df.drop(columns=['Serial', 'serial', 'None', 'none'], errors='ignore', inplace=True)

    return df[['title', 'text', 'label']]


from src.data.loader import load_isot, load_welfake

df_isot = load_isot("Project_IA/True.csv", "Project_IA/Fake.csv")
df_isot.head()

df_welfake = load_welfake("Project_IA/WELFake_Dataset.csv")
df_welfake.head()
