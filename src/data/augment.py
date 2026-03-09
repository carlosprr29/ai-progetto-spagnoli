"Funciones de data augmentation de texto (muy similar al notebook original)"
def shuffle_sentences(text: str, prob: float = 0.5) -> str:
    """
    Con probabilidad dada, mezcla el orden de las oraciones en 'text'.
    Si hay menos de 3 oraciones, devuelve el texto original.
    """
def synonym_replacement(text: str, n: int = 1) -> str:
    """
    Reemplaza aleatoriamente `n` palabras por sinónimos (usa WordNet u otro léxico).
    Control: ignora palabras muy comunes o nombres propios.
    """
from src.data.augment import shuffle_sentences, synonym_replacement

df['text_aug'] = df['text'].apply(lambda x: shuffle_sentences(x, prob=0.3))
df['text_aug'] = df['text_aug'].apply(lambda x: synonym_replacement(x, n=2))
