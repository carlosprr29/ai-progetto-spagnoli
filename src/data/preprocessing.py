"Funciones de limpieza de texto y manejo de datos faltantes"
def remove_reuters_signature(text: str) -> str:
    """
    Elimina la firma de agencia Reuters al inicio del texto, p.ej. "LONDON (Reuters) - ...".
    Usa regex para borrar patrones como '(<location> (Reuters) - )'.
    """
def clean_text(text: str) -> str:
    """
    Normaliza el texto: elimina saltos de línea, URLs, caracteres especiales sobrantes y espacios excesivos.
    Convierte a minúsculas y strip.
    """
def drop_missing(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Elimina filas con valores NA en las columnas indicadas.
    """
def deduplicate(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """
    Elimina filas duplicadas según las columnas indicadas (p.ej. ['title','text']).
    """
from src.data.preprocessing import remove_reuters_signature, clean_text, drop_missing

df = load_isot("Project_IA/True.csv", "Project_IA/Fake.csv")
df['text'] = df['text'].apply(remove_reuters_signature).apply(clean_text)
df = drop_missing(df, ['title','text'])
df = deduplicate(df, ['title','text'])
