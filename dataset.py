"""
Loads the names dataset and builds the character <-> index vocabulary
shared by both the frequency-based and neural bigram models.

A single special token '.' marks both the start and the end of a name
(27 symbols total: 26 letters + '.').
"""

from pathlib import Path


def load_words(path: str = "names.txt") -> list[str]:
    """Read the names dataset, one name per line."""
    return Path(path).read_text(encoding="utf-8").splitlines()


def build_vocab(words: list[str]) -> tuple[dict, dict]:
    """Build stoi (char -> index) / itos (index -> char) mappings.

    '.' is reserved as index 0 and used as both the start and end token,
    so a name like "emma" is modeled as the character sequence
    ['.', 'e', 'm', 'm', 'a', '.'].
    """
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def bigrams(word: str):
    """Yield the (prev_char, next_char) bigrams of a name, including
    the leading/trailing '.' boundary tokens.

    e.g. bigrams("emma") -> ('.','e') ('e','m') ('m','m') ('m','a') ('a','.')
    """
    chars = ["."] + list(word) + ["."]
    yield from zip(chars, chars[1:])
