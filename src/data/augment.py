import random
import nltk
from nltk.corpus import wordnet


# Asegurar que WordNet está disponible
try:
    wordnet.synsets("test")
except LookupError:
    nltk.download("wordnet")


def shuffle_sentences(text: str, prob: float = 0.5) -> str:
    """
    Mezcla aleatoriamente el orden de las oraciones del texto con una
    probabilidad dada.

    Args:
        text: Texto original.
        prob: Probabilidad de aplicar el shuffle.

    Returns:
        Texto con oraciones mezcladas o el original si no se aplica.
    """

    if not text or random.random() > prob:
        return text

    sentences = text.split(". ")

    if len(sentences) < 3:
        return text

    random.shuffle(sentences)

    return ". ".join(sentences)


def synonym_replacement(text: str, n: int = 1) -> str:
    """
    Reemplaza aleatoriamente `n` palabras por sinónimos usando WordNet.

    Args:
        text: Texto original.
        n: Número de palabras a sustituir.

    Returns:
        Texto modificado con sinónimos.
    """

    if not text:
        return text

    words = text.split()
    new_words = words.copy()

    for _ in range(n):

        idx = random.randrange(len(words))
        word = words[idx]

        synsets = wordnet.synsets(word)

        if not synsets:
            continue

        lemmas = [
            lemma.name()
            for lemma in synsets[0].lemmas()
            if lemma.name().lower() != word.lower()
        ]

        if lemmas:
            new_word = random.choice(lemmas).replace("_", " ")
            new_words[idx] = new_word

    return " ".join(new_words)

from src.data.augment import shuffle_sentences, synonym_replacement

df['text_aug'] = df['text'].apply(lambda x: shuffle_sentences(x, prob=0.3))

df['text_aug'] = df['text_aug'].apply(
    lambda x: synonym_replacement(x, n=2)
)
