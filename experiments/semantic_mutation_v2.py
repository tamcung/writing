#!/usr/bin/env python3
"""Semantic-preserving SQLi mutations aimed at breaking shallow surface patterns."""

from __future__ import annotations

import random
import re
from typing import Callable

import sqlparse
from sqlparse import tokens as T

from experiments.consistency_sqli_experiment import (
    append_comment_noise,
    mutate_keywords,
    mutate_separators,
)


SEMANTIC_FAMILIES = {
    "surface_obfuscation": [
        "surface_case",
        "surface_space_comment",
        "surface_comment_tail",
    ],
    "numeric_repr": [
        "numeric_hex",
        "numeric_select",
        "numeric_arithmetic",
    ],
    "string_construction": [
        "string_char_codes",
        "string_concat_split",
        "string_hex_literal",
        "string_unhex_literal",
        "string_concat_mixed",
    ],
    "boolean_equivalent": [
        "boolean_true_equivalent",
    ],
}


def _flatten_sql(text: str):
    tokens = []
    try:
        for stmt in sqlparse.parse(text):
            tokens.extend(list(stmt.flatten()))
    except Exception:
        return []
    return tokens


def _safe_single_quoted_inner(value: str) -> str | None:
    if len(value) < 2 or not (value.startswith("'") and value.endswith("'")):
        return None
    inner = value[1:-1]
    if "\\" in inner or "'" in inner:
        return None
    return inner


def _safe_quoted_inner(value: str) -> str | None:
    if len(value) < 2:
        return None
    quote = value[0]
    if quote not in {"'", '"'} or value[-1] != quote:
        return None
    inner = value[1:-1]
    if "\\" in inner or quote in inner:
        return None
    return inner


def _is_integer_token(token) -> bool:
    return token.ttype is not None and token.ttype in T.Literal.Number.Integer


def _is_single_string_token(token) -> bool:
    if token.ttype is not None and token.ttype in T.Literal.String.Single:
        return True
    return _safe_quoted_inner(token.value) is not None


def surface_case(text: str, rng: random.Random) -> str:
    return mutate_keywords(text, rng)


def surface_space_comment(text: str, rng: random.Random) -> str:
    return mutate_separators(text, rng, hard=True)


def surface_comment_tail(text: str, rng: random.Random) -> str:
    return append_comment_noise(text, rng)


def numeric_hex(text: str, rng: random.Random) -> str:
    parsed = _flatten_sql(text)
    if not parsed:
        return text
    out = []
    changed = False
    for token in parsed:
        value = token.value
        if _is_integer_token(token) and not value.lower().startswith("0x"):
            intval = int(value)
            if intval >= 0:
                out.append(f"0x{intval:x}")
                changed = True
            else:
                out.append(value)
        else:
            out.append(value)
    return "".join(out) if changed else text


def numeric_select(text: str, rng: random.Random) -> str:
    parsed = _flatten_sql(text)
    if not parsed:
        return text
    out = []
    changed = False
    for token in parsed:
        value = token.value
        if _is_integer_token(token) and not value.lower().startswith("0x"):
            out.append(f"(SELECT {value})")
            changed = True
        else:
            out.append(value)
    return "".join(out) if changed else text


def numeric_arithmetic(text: str, rng: random.Random) -> str:
    parsed = _flatten_sql(text)
    if not parsed:
        return text
    variants = [
        lambda v: f"({v}+0)",
        lambda v: f"({v}-0)",
        lambda v: f"({v}*1)",
    ]
    out = []
    changed = False
    for token in parsed:
        value = token.value
        if _is_integer_token(token) and not value.lower().startswith("0x"):
            out.append(rng.choice(variants)(value))
            changed = True
        else:
            out.append(value)
    return "".join(out) if changed else text


def string_char_codes(text: str, rng: random.Random) -> str:
    parsed = _flatten_sql(text)
    if not parsed:
        return text
    out = []
    changed = False
    for token in parsed:
        value = token.value
        inner = _safe_quoted_inner(value) if _is_single_string_token(token) else None
        if inner and 0 < len(inner) <= 12 and all(32 <= ord(ch) <= 126 for ch in inner):
            codes = ",".join(str(ord(ch)) for ch in inner)
            out.append(f"CHAR({codes})")
            changed = True
        else:
            out.append(value)
    return "".join(out) if changed else text


def string_concat_split(text: str, rng: random.Random) -> str:
    parsed = _flatten_sql(text)
    if not parsed:
        return text
    out = []
    changed = False
    for token in parsed:
        value = token.value
        inner = _safe_quoted_inner(value) if _is_single_string_token(token) else None
        if inner and 2 <= len(inner) <= 16:
            if len(inner) == 2:
                cuts = [1]
            else:
                first_cut = max(1, min(len(inner) - 1, len(inner) // 2))
                cuts = [first_cut]
                if len(inner) >= 6 and rng.random() < 0.5:
                    second_cut = max(first_cut + 1, min(len(inner) - 1, (2 * len(inner)) // 3))
                    if second_cut < len(inner):
                        cuts.append(second_cut)
            parts = []
            start = 0
            for cut in cuts:
                parts.append(inner[start:cut])
                start = cut
            parts.append(inner[start:])
            if len(parts) >= 2 and all(part for part in parts):
                quoted = ",".join(f"'{part}'" for part in parts)
                out.append(f"CONCAT({quoted})")
                changed = True
            else:
                out.append(value)
        else:
            out.append(value)
    return "".join(out) if changed else text


def string_hex_literal(text: str, rng: random.Random) -> str:
    parsed = _flatten_sql(text)
    if not parsed:
        return text
    out = []
    changed = False
    for token in parsed:
        value = token.value
        inner = _safe_quoted_inner(value) if _is_single_string_token(token) else None
        if inner and 0 < len(inner) <= 16 and all(32 <= ord(ch) <= 126 for ch in inner):
            out.append("0x" + inner.encode("utf-8").hex())
            changed = True
        else:
            out.append(value)
    return "".join(out) if changed else text


def string_unhex_literal(text: str, rng: random.Random) -> str:
    parsed = _flatten_sql(text)
    if not parsed:
        return text
    out = []
    changed = False
    for token in parsed:
        value = token.value
        inner = _safe_quoted_inner(value) if _is_single_string_token(token) else None
        if inner and 0 < len(inner) <= 16 and all(32 <= ord(ch) <= 126 for ch in inner):
            out.append(f"UNHEX('{inner.encode('utf-8').hex()}')")
            changed = True
        else:
            out.append(value)
    return "".join(out) if changed else text


def string_concat_mixed(text: str, rng: random.Random) -> str:
    parsed = _flatten_sql(text)
    if not parsed:
        return text
    out = []
    changed = False
    for token in parsed:
        value = token.value
        inner = _safe_quoted_inner(value) if _is_single_string_token(token) else None
        if inner and 3 <= len(inner) <= 18 and all(32 <= ord(ch) <= 126 for ch in inner):
            parts = []
            idx = 0
            while idx < len(inner):
                step = 1 if rng.random() < 0.45 else 2
                chunk = inner[idx : idx + step]
                if len(chunk) == 1:
                    style = rng.choice(["char", "hex", "plain"])
                    if style == "char":
                        parts.append(f"CHAR({ord(chunk)})")
                    elif style == "hex":
                        parts.append("0x" + chunk.encode("utf-8").hex())
                    else:
                        parts.append(f"'{chunk}'")
                else:
                    style = rng.choice(["plain", "hex"])
                    if style == "hex":
                        parts.append("0x" + chunk.encode("utf-8").hex())
                    else:
                        parts.append(f"'{chunk}'")
                idx += step
            if len(parts) >= 2:
                out.append(f"CONCAT({','.join(parts)})")
                changed = True
            else:
                out.append(value)
        else:
            out.append(value)
    return "".join(out) if changed else text


TRUE_EQUIV_PATTERNS = [
    re.compile(r"\b1\s*=\s*1\b", re.I),
    re.compile(r"'a'\s*=\s*'a'", re.I),
    re.compile(r"'1'\s*=\s*'1'", re.I),
]

TRUE_EQUIV_REPLACEMENTS = [
    "TRUE",
    "NOT FALSE",
    "1 BETWEEN 1 AND 1",
    "STRCMP('a','a')=0",
    "(SELECT 1)=(SELECT 1)",
]


def boolean_true_equivalent(text: str, rng: random.Random) -> str:
    matches = []
    for pat in TRUE_EQUIV_PATTERNS:
        matches.extend(list(pat.finditer(text)))
    if not matches:
        return text
    match = rng.choice(matches)
    replacement = rng.choice(TRUE_EQUIV_REPLACEMENTS)
    return text[: match.start()] + replacement + text[match.end() :]


SEMANTIC_STRATEGIES: dict[str, Callable[[str, random.Random], str]] = {
    "surface_case": surface_case,
    "surface_space_comment": surface_space_comment,
    "surface_comment_tail": surface_comment_tail,
    "numeric_hex": numeric_hex,
    "numeric_select": numeric_select,
    "numeric_arithmetic": numeric_arithmetic,
    "string_char_codes": string_char_codes,
    "string_concat_split": string_concat_split,
    "string_hex_literal": string_hex_literal,
    "string_unhex_literal": string_unhex_literal,
    "string_concat_mixed": string_concat_mixed,
    "boolean_true_equivalent": boolean_true_equivalent,
}


def apply_semantic_strategy_rounds(
    text: str,
    strategy_names: list[str],
    seed: int,
    rounds: int,
    ensure_changed: bool,
    retries: int,
) -> str:
    last = text
    for attempt in range(max(1, retries)):
        rng = random.Random(seed + attempt * 104_729)
        out = text
        for _ in range(max(1, rounds)):
            name = rng.choice(strategy_names)
            out = SEMANTIC_STRATEGIES[name](out, rng)
        last = out
        if not ensure_changed or out != text:
            return out[:260]
    return last[:260]


def build_semantic_test_family_view(
    test_texts: list[str],
    test_labels: list[int],
    seed: int,
    strategy_names: list[str],
    rounds: int,
    retries: int,
) -> tuple[list[str], list[int], list[str], list[str]]:
    texts = []
    base_sqli = []
    aug_sqli = []
    for i, (text, y) in enumerate(zip(test_texts, test_labels)):
        if y == 1:
            aug = apply_semantic_strategy_rounds(
                text=text,
                strategy_names=strategy_names,
                seed=seed * 100_003 + i,
                rounds=rounds,
                ensure_changed=True,
                retries=retries,
            )
            texts.append(aug)
            base_sqli.append(text)
            aug_sqli.append(aug)
        else:
            texts.append(text)
    return texts, list(test_labels), base_sqli, aug_sqli
