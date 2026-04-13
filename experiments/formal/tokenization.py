#!/usr/bin/env python3
"""Tokenization helpers for SQLi-like strings."""

from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(
    r"""
    /\*.*?\*/              |
    --[^\r\n]*             |
    %[0-9a-fA-F]{2}        |
    0x[0-9a-fA-F]+         |
    [A-Za-z_][A-Za-z_0-9]* |
    \d+(?:\.\d+)?          |
    <>|!=|<=|>=|==|\|\||&& |
    \S
    """,
    re.S | re.X,
)


def tokenize_sql(text: str, lowercase: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if lowercase:
        tokens = [tok.lower() for tok in tokens]
    return tokens or ["<EMPTY>"]


def build_vocab(texts: list[str], max_vocab: int = 20000, min_freq: int = 1, lowercase: bool = True) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize_sql(text, lowercase=lowercase))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counts.most_common(max(0, max_vocab - len(vocab))):
        if freq < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(text: str, vocab: dict[str, int], max_tokens: int, lowercase: bool = True) -> list[int]:
    ids = [vocab.get(tok, 1) for tok in tokenize_sql(text, lowercase=lowercase)[:max_tokens]]
    if len(ids) < max_tokens:
        ids.extend([0] * (max_tokens - len(ids)))
    return ids
