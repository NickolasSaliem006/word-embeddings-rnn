"""Provided text processing. There are no student TODOs in this file."""

import html
import re
import unicodedata
from collections import Counter

PAD_ID, UNK_ID, NUM_ID = 0, 1, 2
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<NUM>"]


def tokenize(text: str) -> list[str]:
    """Return lowercase words/punctuation; numbers become a shared NUM token."""
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("\u2019", "'").lower()
    tokens = re.findall(r"\d+(?:\.\d+)?|\w+(?:'\w+)*|[^\w\s]", text)
    return ["<NUM>" if re.fullmatch(r"\d+(?:\.\d+)?", word) else word for word in tokens]


def build_vocab(documents: list[list[str]], max_size: int = 10_000) -> dict[str, int]:
    """Build a deterministic word-to-ID table from training texts only."""
    counts = Counter(word for document in documents for word in document)
    words = [word for word, count in counts.items() if count >= 2 and word not in SPECIAL_TOKENS]
    words.sort(key=lambda word: (-counts[word], word))
    return {word: index for index, word in enumerate(SPECIAL_TOKENS + words[: max_size - 3])}


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """Map unknown words to UNK. Empty documents get one UNK token."""
    return [vocab.get(word, UNK_ID) for word in tokens] or [UNK_ID]
