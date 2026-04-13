#!/usr/bin/env python3
"""Formal semantic-preserving mutation families for SQLi robustness experiments."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable


MutationFn = Callable[[str, random.Random], str]


SQL_KEYWORDS = [
    "select",
    "union",
    "from",
    "where",
    "and",
    "or",
    "sleep",
    "benchmark",
    "concat",
    "char",
    "hex",
    "substr",
    "substring",
    "ascii",
    "mid",
    "if",
    "case",
    "when",
    "then",
    "else",
    "end",
    "null",
    "order",
    "by",
    "group",
    "having",
]

SURFACE_OBFUSCATION = "surface_obfuscation"
NUMERIC_REPR = "numeric_repr"
STRING_CONSTRUCTION = "string_construction"
BOOLEAN_EQUIVALENT = "boolean_equivalent"

PRIMARY_FAMILIES = [SURFACE_OBFUSCATION, NUMERIC_REPR, STRING_CONSTRUCTION]
ALL_FAMILIES = PRIMARY_FAMILIES + [BOOLEAN_EQUIVALENT]


def _rewrite_keywords(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        mode = rng.choice(["upper", "lower", "title", "mixed"])
        if mode == "upper":
            return token.upper()
        if mode == "lower":
            return token.lower()
        if mode == "title":
            return token.title()
        return "".join(ch.upper() if i % 2 == 0 else ch.lower() for i, ch in enumerate(token))

    pattern = r"\b(" + "|".join(re.escape(k) for k in SQL_KEYWORDS) + r")\b"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def _rewrite_spaces(text: str, rng: random.Random) -> str:
    styles = [" ", "/**/", "\t", "\n", "\f", "\v"]

    def repl(match: re.Match[str]) -> str:
        return rng.choice(styles)

    return re.sub(r"\s+", repl, text)


def mutate_surface_obfuscation(text: str, rng: random.Random) -> str:
    mutated = _rewrite_keywords(text, rng)
    mutated = _rewrite_spaces(mutated, rng)
    return mutated


def mutate_numeric_repr(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        if len(raw) > 6:
            return raw
        try:
            value = int(raw)
        except ValueError:
            return raw
        mode = rng.choice(["hex", "paren", "pluszero"])
        if mode == "hex":
            return hex(value)
        if mode == "paren":
            return f"({value})"
        return f"({value}+0)"

    mutated = re.sub(r"(?<![%A-Za-z0-9_])\d+(?![A-Za-z0-9_])", repl, text)
    if mutated == text:
        mutated = text.replace("1=1", "0x1=0x1")
        mutated = mutated.replace("1 = 1", "0x1 = 0x1")
    return mutated


def _char_encode(content: str) -> str:
    if not content:
        return "''"
    return "concat(" + ",".join(f"char({ord(ch)})" for ch in content) + ")"


def _quoted_concat_encode(content: str, quote: str, rng: random.Random) -> str:
    if len(content) <= 1:
        return f"{quote}{content}{quote}"
    split_at = rng.randint(1, len(content) - 1)
    left = content[:split_at]
    right = content[split_at:]
    return f"concat({quote}{left}{quote},{quote}{right}{quote})"


def _safe_ascii_from_hex(hex_text: str) -> str | None:
    if len(hex_text) % 2 != 0:
        return None
    try:
        raw = bytes.fromhex(hex_text)
    except ValueError:
        return None
    if not raw:
        return None
    if len(raw) > 12:
        return None
    if any(b < 32 or b > 126 for b in raw):
        return None
    return raw.decode("ascii")


def mutate_string_construction(text: str, rng: random.Random) -> str:
    def single_quote_repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        if len(inner) > 16:
            return match.group(0)
        mode = rng.choice(["char", "split_concat"])
        if mode == "char":
            return _char_encode(inner)
        return _quoted_concat_encode(inner, "'", rng)

    mutated = re.sub(r"(?<![A-Za-z0-9_])'([A-Za-z0-9_]{1,16})'(?!')", single_quote_repl, text)
    if mutated != text:
        return mutated

    def double_quote_repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        if len(inner) > 16:
            return match.group(0)
        # Payload datasets often contain double-quoted string fragments used as literals.
        mode = rng.choice(["char", "split_concat"])
        if mode == "char":
            return _char_encode(inner)
        return _quoted_concat_encode(inner, '"', rng)

    mutated = re.sub(r'(?<![A-Za-z0-9_])"([A-Za-z0-9_]{1,16})"(?!")', double_quote_repl, text)
    if mutated != text:
        return mutated

    def hex_repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        decoded = _safe_ascii_from_hex(inner)
        if decoded is None:
            return match.group(0)
        return _char_encode(decoded)

    mutated = re.sub(r"\b0x([0-9a-fA-F]{2,24})\b", hex_repl, text)
    if mutated != text:
        return mutated

    return mutated


def mutate_boolean_equivalent(text: str, rng: random.Random) -> str:
    replacements = [
        ("1=1", "(2>1)"),
        ("1 = 1", "(2 > 1)"),
        ("0=0", "(3>2)"),
        ("0 = 0", "(3 > 2)"),
        ("true", "(1=1)"),
        ("false", "(1=0)"),
    ]
    mutated = text
    rng.shuffle(replacements)
    for src, dst in replacements:
        if src in mutated:
            mutated = mutated.replace(src, dst, 1)
            return mutated
    if re.search(r"\bor\b", mutated, flags=re.IGNORECASE):
        return re.sub(r"\bor\b", "OR (2>1) AND", mutated, count=1, flags=re.IGNORECASE)
    return mutated


FAMILY_TO_FN: dict[str, MutationFn] = {
    SURFACE_OBFUSCATION: mutate_surface_obfuscation,
    NUMERIC_REPR: mutate_numeric_repr,
    STRING_CONSTRUCTION: mutate_string_construction,
    BOOLEAN_EQUIVALENT: mutate_boolean_equivalent,
}


@dataclass(frozen=True)
class MutationRecord:
    source_text: str
    mutated_text: str
    family: str
    rounds: int


def mutate_with_family(text: str, family: str, seed: int, rounds: int = 1) -> MutationRecord:
    if family not in FAMILY_TO_FN:
        raise KeyError(f"unknown mutation family: {family}")
    rng = random.Random(seed)
    current = text
    fn = FAMILY_TO_FN[family]
    for _ in range(max(1, rounds)):
        current = fn(current, rng)
    return MutationRecord(source_text=text, mutated_text=current, family=family, rounds=max(1, rounds))


def mutate_with_mixed_families(text: str, families: list[str], seed: int, rounds: int = 3) -> MutationRecord:
    if not families:
        raise ValueError("families must not be empty")
    rng = random.Random(seed)
    current = text
    chosen: list[str] = []
    for _ in range(max(1, rounds)):
        family = rng.choice(families)
        chosen.append(family)
        current = FAMILY_TO_FN[family](current, rng)
    return MutationRecord(
        source_text=text,
        mutated_text=current,
        family="+".join(chosen),
        rounds=max(1, rounds),
    )


def mutate_with_forced_surface_mixed(text: str, seed: int, rounds: int = 3) -> MutationRecord:
    """Apply value-level rewrites first and a SQL-level surface rewrite last.

    The final surface pass is intentional: random mixed mutations can repeatedly
    pick numeric rewrites and leave lexical SQL keywords untouched, which makes
    the hard view too weak for an obfuscation-robustness stress test.
    """

    rng = random.Random(seed)
    current = text
    chosen: list[str] = []
    value_families = [NUMERIC_REPR, STRING_CONSTRUCTION]

    for _ in range(max(0, rounds - 1)):
        family = rng.choice(value_families)
        mutated = FAMILY_TO_FN[family](current, rng)
        if mutated != current:
            current = mutated
            chosen.append(family)

    current = mutate_surface_obfuscation(current, rng)
    chosen.append(SURFACE_OBFUSCATION)

    return MutationRecord(
        source_text=text,
        mutated_text=current,
        family="+".join(chosen),
        rounds=len(chosen),
    )


def benign_nuisance_transform(text: str, seed: int) -> str:
    rng = random.Random(seed)
    current = text
    if rng.random() < 0.5:
        current = re.sub(r"\s+", lambda _: rng.choice([" ", "\t", "\n"]), current)
    if rng.random() < 0.5:
        current = _rewrite_keywords(current, rng)
    return current
