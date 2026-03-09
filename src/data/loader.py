"Funciones para leer CSVs y unirlos"
def load_isot(true_path: str, fake_path: str) -> pd.DataFrame:
    """
    Carga los CSV de ISOT (noticias reales y falsas) y devuelve un DataFrame unificado.
    
    Args:
        true_path: Ruta al CSV de noticias reales.
        fake_path: Ruta al CSV de noticias falsas.
    Returns:
        DataFrame con columnas ['title','text','label'], donde label 0=real,1=fake.
    """
def load_welfake(path: str) -> pd.DataFrame:
    """
    Carga el CSV WELFake y devuelve un DataFrame con columnas esenciales.
    
    Args:
        path: Ruta al CSV WELFake.
    Returns:
        DataFrame con ['title','text','label'].
    """
from src.data.loader import load_isot, load_welfake

df_isot = load_isot("Project_IA/True.csv", "Project_IA/Fake.csv")
df_isot.head()

df_welfake = load_welfake("Project_IA/WELFake_Dataset.csv")
