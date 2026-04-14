#!/usr/bin/env python3
"""Targeted SQL-level semantic mutations for adversarial SQLi evaluation."""

from __future__ import annotations

import random
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


MutationFn = Callable[[str, random.Random], str]
ScoreFn = Callable[[list[str]], np.ndarray]
# Returns per-token importance scores (aligned with tokenize_sql output), or None for non-differentiable models.
GradFn = Callable[[str], "np.ndarray | None"]


SQL_KEYWORDS = [    "select",
    "union",
    "all",
    "distinct",
    "from",
    "where",
    "and",
    "or",
    "like",
    "not",
    "sleep",
    "benchmark",
    "concat",
    "char",
    "hex",
    "unhex",
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
    "limit",
]

_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in SQL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

_SQL_KEYWORDS_SET = frozenset(k.lower() for k in SQL_KEYWORDS)

# Map operator family → which token class it primarily targets.
# "whitespace" operators target the gaps between tokens, so we use all-token mean importance.
_FAMILY_TO_TOKEN_CLASS: dict[str, str] = {
    "surface_obfuscation": "whitespace",
    "numeric_repr": "number",
    "string_construction": "string",
    "boolean_equivalent": "boolean",
    "operator_synonym": "operator",
    "comment_marker": "comment",
    "mysql_comment": "keyword",
    "official_wafamole": "other",
}

# Operator-name overrides that are more specific than the family-level mapping.
_OP_NAME_TO_TOKEN_CLASS: dict[str, str] = {
    "random_case_keywords": "keyword",
    "reset_inline_comments": "comment",
    "comment_rewriting": "comment",
    "mysql_comment_marker_synonym": "comment",
}


def _classify_token(tok: str) -> str:
    tl = tok.lower()
    if tl in _SQL_KEYWORDS_SET:
        return "keyword"
    if re.fullmatch(r"\d+(?:\.\d+)?|0x[0-9a-fA-F]+", tok, re.IGNORECASE):
        return "number"
    if tok.startswith("/*") or tok.startswith("--") or tok.startswith("#"):
        return "comment"
    if re.fullmatch(r"['\"][A-Za-z0-9_@$#]*['\"]|['\"]", tok):
        return "string"
    if tok in ("||", "&&", "!=", "<>", "=", "<", ">", "<=", ">=", "==", "LIKE", "NOT"):
        return "operator"
    return "other"


def _operator_weights_from_importances(
    operators: list["SqlMutationOperator"],
    tokens: list[str],
    importances: np.ndarray,
    floor: float = 0.05,
) -> list[float]:
    """Compute sampling weight per operator based on gradient importance of the token types it targets."""
    # Aggregate importance by token class
    class_imp: dict[str, float] = {}
    for tok, imp in zip(tokens, importances[: len(tokens)]):
        cls = _classify_token(tok)
        class_imp[cls] = class_imp.get(cls, 0.0) + float(imp)

    if not class_imp:
        return [1.0] * len(operators)

    total = sum(class_imp.values())
    mean_imp = total / len(class_imp)

    weights: list[float] = []
    for op in operators:
        target_cls = _OP_NAME_TO_TOKEN_CLASS.get(op.name) or _FAMILY_TO_TOKEN_CLASS.get(op.family, "other")
        if target_cls == "whitespace":
            # Whitespace operators affect the overall sequence structure;
            # proxy = mean importance across all tokens.
            w = mean_imp
        else:
            w = class_imp.get(target_cls, mean_imp)
        weights.append(max(w, floor * mean_imp))
    return weights


@dataclass(frozen=True)
class SqlMutationOperator:
    name: str
    family: str
    semantic_note: str
    fn: MutationFn


@dataclass(frozen=True)
class CandidateState:
    text: str
    prob: float
    chain: tuple[str, ...]


@dataclass(frozen=True)
class TargetedSearchResult:
    source_text: str
    adversarial_text: str
    source_prob: float
    adversarial_prob: float
    success: bool
    changed: bool
    steps: int
    queries: int
    chain: tuple[str, ...]
    history: tuple[dict, ...]


@dataclass
class _SearchSession:
    source_text: str
    source_prob: float
    rng: random.Random
    best: CandidateState
    beam: list[CandidateState]
    visited_texts: set[str]
    queries: int
    history: list[dict]
    done: bool = False


@dataclass(frozen=True)
class RandomMutationResult:
    source_text: str
    mutated_text: str
    changed: bool
    chain: tuple[str, ...]


_PATCHED_RANDOM_FUNCS = (
    "choice",
    "choices",
    "randint",
    "randrange",
    "random",
    "sample",
    "shuffle",
    "uniform",
    "getrandbits",
)


def _call_with_rng(fn: Callable[[str], str], text: str, rng: random.Random) -> str:
    saved = {name: getattr(random, name) for name in _PATCHED_RANDOM_FUNCS}
    try:
        for name in _PATCHED_RANDOM_FUNCS:
            setattr(random, name, getattr(rng, name))
        return fn(text)
    finally:
        for name, original in saved.items():
            setattr(random, name, original)


def _quote_mask(text: str) -> list[bool]:
    mask = [False] * len(text)
    quote: str | None = None
    escaped = False
    quote_start: int | None = None
    for i, ch in enumerate(text):
        if quote is not None:
            mask[i] = True
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
                quote_start = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            quote_start = i
            mask[i] = True
    if quote is not None and quote_start is not None:
        for i in range(quote_start, len(text)):
            mask[i] = False
    return mask


def _outside_quotes(mask: list[bool], start: int, end: int) -> bool:
    return not any(mask[start:end])


def _finditer_outside_quotes(pattern: str, text: str, flags: int = 0) -> list[re.Match[str]]:
    mask = _quote_mask(text)
    return [m for m in re.finditer(pattern, text, flags) if _outside_quotes(mask, m.start(), m.end())]


def _replace_span(text: str, span: tuple[int, int], replacement: str) -> str:
    return text[: span[0]] + replacement + text[span[1] :]


def _replace_random_match(
    text: str,
    pattern: str,
    replacement_fn: Callable[[re.Match[str]], str],
    rng: random.Random,
    flags: int = 0,
    outside_quotes: bool = True,
) -> str:
    matches = _finditer_outside_quotes(pattern, text, flags) if outside_quotes else list(re.finditer(pattern, text, flags))
    if not matches:
        return text
    match = rng.choice(matches)
    return _replace_span(text, match.span(), replacement_fn(match))


def _random_ascii_word(rng: random.Random, min_len: int = 1, max_len: int = 5) -> str:
    size = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(size))


def _randomize_case(token: str, rng: random.Random) -> str:
    return "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in token)


def op_random_case_keywords(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        mutated = _randomize_case(token, rng)
        return mutated if mutated != token else token.swapcase()

    return _KEYWORD_RE.sub(repl, text)


def op_spaces_to_comments(text: str, rng: random.Random) -> str:
    return _replace_random_match(text, r"\s+", lambda _: rng.choice(["/**/", "/*_*/", "/*0*/"]), rng)


def op_spaces_to_alternatives(text: str, rng: random.Random) -> str:
    alternatives = ["\t", "\n", "\f", "\v", "  "]
    return _replace_random_match(text, r"\s+", lambda _: rng.choice(alternatives), rng)


def op_many_separators(text: str, rng: random.Random) -> str:
    mask = _quote_mask(text)
    out: list[str] = []
    i = 0
    changed = False
    while i < len(text):
        match = re.match(r"\s+", text[i:])
        if match and _outside_quotes(mask, i, i + len(match.group(0))):
            repl = rng.choice(["/**/", "\t", "\n", "/*_*/"])
            out.append(repl)
            changed = changed or repl != match.group(0)
            i += len(match.group(0))
            continue
        out.append(text[i])
        i += 1
    return "".join(out) if changed else text


def op_reset_inline_comments(text: str, rng: random.Random) -> str:
    return _replace_random_match(text, r"/\*(?!\!)[^*]*\*/", lambda _: "/**/", rng, outside_quotes=True)


def op_comment_rewriting(text: str, rng: random.Random) -> str:
    if "#" in text or "-- " in text:
        return text + _random_ascii_word(rng, 2, 5)
    return text + "/**/"


def op_integer_hex(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        value = int(match.group(0))
        return hex(value)

    return _replace_random_match(text, r"(?<![%A-Za-z0-9_])\d{1,6}(?![A-Za-z0-9_])", repl, rng)


def op_integer_select(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"(SELECT {match.group(0)})"

    return _replace_random_match(text, r"(?<![%A-Za-z0-9_])\d{1,6}(?![A-Za-z0-9_])", repl, rng)


def op_integer_arithmetic(text: str, rng: random.Random) -> str:
    variants = [
        lambda value: f"({value}+0)",
        lambda value: f"({value}-0)",
        lambda value: f"({value}*1)",
        lambda value: f"({value}/1)",
    ]

    def repl(match: re.Match[str]) -> str:
        return rng.choice(variants)(match.group(0))

    return _replace_random_match(text, r"(?<![%A-Za-z0-9_])\d{1,6}(?![A-Za-z0-9_])", repl, rng)


def _quoted_string_repl(match: re.Match[str], rng: random.Random) -> str:
    quote = match.group(1)
    inner = match.group(2)
    if not inner or len(inner) > 16:
        return match.group(0)
    if any(ord(ch) < 32 or ord(ch) > 126 for ch in inner):
        return match.group(0)
    mode = rng.choice(["char", "concat", "hex"])
    if mode == "char":
        return "CHAR(" + ",".join(str(ord(ch)) for ch in inner) + ")"
    if mode == "hex":
        return "0x" + inner.encode("utf-8").hex()
    if len(inner) == 1:
        return f"CHAR({ord(inner)})"
    cut = rng.randint(1, len(inner) - 1)
    return f"CONCAT({quote}{inner[:cut]}{quote},{quote}{inner[cut:]}{quote})"


def op_string_construction(text: str, rng: random.Random) -> str:
    pattern = r"(?<![A-Za-z0-9_])(['\"])([A-Za-z0-9_@$#]{1,16})\1"
    return _replace_random_match(text, pattern, lambda m: _quoted_string_repl(m, rng), rng, outside_quotes=False)


def op_boolean_tautology_substitution(text: str, rng: random.Random) -> str:
    numeric = r"\b(\d{1,6})\s*(=|LIKE)\s*\1\b"
    matches = _finditer_outside_quotes(numeric, text, flags=re.IGNORECASE)
    if not matches:
        string_pat = r"(['\"])([A-Za-z][A-Za-z0-9_@$#]{0,15})\1\s*(=|LIKE)\s*(['\"])\2\4"
        matches = list(re.finditer(string_pat, text, flags=re.IGNORECASE))
    if not matches:
        return text
    match = rng.choice(matches)
    value = rng.randint(2, 9999)
    token = _random_ascii_word(rng)
    replacements = [
        f"{value}={value}",
        f"{value} BETWEEN {value - 1} AND {value + 1}",
        f"'{token}'='{token}'",
        "(SELECT 1)=(SELECT 1)",
    ]
    replacement = rng.choice(replacements)
    return _replace_span(text, match.span(), replacement)


def op_logical_invariant(text: str, rng: random.Random) -> str:
    patterns = [
        r"\b\d{1,6}\s*(=|LIKE)\s*\d{1,6}\b",
        r"(['\"])[A-Za-z][A-Za-z0-9_@$#]{0,15}\1\s*(=|LIKE)\s*(['\"])[A-Za-z][A-Za-z0-9_@$#]{0,15}\3",
    ]
    matches: list[re.Match[str]] = []
    matches.extend(_finditer_outside_quotes(patterns[0], text, flags=re.IGNORECASE))
    matches.extend(list(re.finditer(patterns[1], text, flags=re.IGNORECASE)))
    if not matches:
        return text
    match = rng.choice(matches)
    suffix = rng.choice(
        [
            " AND 1=1",
            " AND TRUE",
            " AND (SELECT 1)=(SELECT 1)",
            " OR 1=0",
            " OR FALSE",
            " OR (SELECT 1)=(SELECT 0)",
        ]
    )
    return text[: match.end()] + suffix + text[match.end() :]


def op_boolean_tautology_to_literal(text: str, rng: random.Random) -> str:
    patterns = [
        r"\b(\d{1,6})\s*=\s*\1\b",
        r"(['\"])([A-Za-z][A-Za-z0-9_@$#]{0,15})\1\s*=\s*(['\"])\2\3",
    ]
    matches: list[re.Match[str]] = []
    matches.extend(_finditer_outside_quotes(patterns[0], text, flags=re.IGNORECASE))
    matches.extend(list(re.finditer(patterns[1], text, flags=re.IGNORECASE)))
    if not matches:
        return text
    match = rng.choice(matches)
    replacement = rng.choice(["TRUE", "NOT FALSE", "(TRUE)"])
    return _replace_span(text, match.span(), replacement)


def op_mysql_operator_synonym(text: str, rng: random.Random) -> str:
    replacements = [
        (r"\bOR\b", "||"),
        (r"\bAND\b", "&&"),
        (r"<>", "!="),
        (r"!=", "<>"),
    ]
    rng.shuffle(replacements)
    for pattern, replacement in replacements:
        mutated = _replace_random_match(text, pattern, lambda _: replacement, rng, flags=re.IGNORECASE)
        if mutated != text:
            return mutated
    return text


def op_mysql_tautology_like(text: str, rng: random.Random) -> str:
    patterns = [
        r"\b(\d{1,6})\s*=\s*\1\b",
        r"(['\"])([A-Za-z][A-Za-z0-9_@$#]{0,15})\1\s*=\s*(['\"])\2\3",
    ]
    matches: list[re.Match[str]] = []
    matches.extend(_finditer_outside_quotes(patterns[0], text, flags=re.IGNORECASE))
    matches.extend(list(re.finditer(patterns[1], text, flags=re.IGNORECASE)))
    if not matches:
        return text
    match = rng.choice(matches)
    replacement = match.group(0).replace("=", " LIKE ", 1)
    return _replace_span(text, match.span(), replacement)


def op_mysql_comment_marker_synonym(text: str, rng: random.Random) -> str:
    del rng
    if "--" in text:
        return text.replace("--", "#", 1)
    if "#" in text:
        return text.replace("#", "-- ", 1)
    return text


def op_mysql_executable_comment_keyword(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0).upper()
        return f"/*!50000{token}*/"

    return _replace_random_match(text, _KEYWORD_RE.pattern, repl, rng, flags=re.IGNORECASE)


def op_many_mysql_executable_comments(text: str, rng: random.Random) -> str:
    del rng

    def repl(match: re.Match[str]) -> str:
        return f"/*!50000{match.group(0).upper()}*/"

    return _KEYWORD_RE.sub(repl, text)


CONSERVATIVE_OPERATORS = [
    SqlMutationOperator("random_case_keywords", "surface_obfuscation", "SQL keywords are case-insensitive.", op_random_case_keywords),
    SqlMutationOperator("spaces_to_comments", "surface_obfuscation", "SQL comments separate tokens without changing token sequence.", op_spaces_to_comments),
    SqlMutationOperator("spaces_to_alternatives", "surface_obfuscation", "SQL treats common whitespace characters as separators.", op_spaces_to_alternatives),
    SqlMutationOperator("many_separators", "surface_obfuscation", "Multiple token separators are rewritten without changing token order.", op_many_separators),
    SqlMutationOperator("reset_inline_comments", "surface_obfuscation", "Comment payload is ignored by SQL parsers.", op_reset_inline_comments),
    SqlMutationOperator("comment_rewriting", "surface_obfuscation", "Trailing/comment content is ignored by SQL parsers.", op_comment_rewriting),
    SqlMutationOperator("integer_hex", "numeric_repr", "Positive integer constants are rewritten as hexadecimal constants.", op_integer_hex),
    SqlMutationOperator("integer_select", "numeric_repr", "A scalar SELECT returning the same integer is value-equivalent.", op_integer_select),
    SqlMutationOperator("integer_arithmetic", "numeric_repr", "Arithmetic identities preserve integer values.", op_integer_arithmetic),
    SqlMutationOperator("string_construction", "string_construction", "Short quoted ASCII literals are rebuilt with CHAR/CONCAT/hex forms.", op_string_construction),
    SqlMutationOperator("boolean_tautology_substitution", "boolean_equivalent", "A matched tautology is replaced by another tautology.", op_boolean_tautology_substitution),
    SqlMutationOperator("logical_invariant", "boolean_equivalent", "Appending AND TRUE or OR FALSE to a boolean expression preserves truth value.", op_logical_invariant),
    SqlMutationOperator("boolean_tautology_to_literal", "boolean_equivalent", "A matched tautology is replaced with TRUE or NOT FALSE.", op_boolean_tautology_to_literal),
]

WAFAMOLE_STYLE_OPERATORS = CONSERVATIVE_OPERATORS + [
    SqlMutationOperator("mysql_operator_synonym", "operator_synonym", "MySQL-style operator synonyms such as OR/|| and AND/&&.", op_mysql_operator_synonym),
    SqlMutationOperator("mysql_tautology_like", "operator_synonym", "MySQL-style LIKE is used on matched literal tautologies.", op_mysql_tautology_like),
    SqlMutationOperator("mysql_comment_marker_synonym", "comment_marker", "MySQL comment marker synonyms rewrite -- and #.", op_mysql_comment_marker_synonym),
    SqlMutationOperator("mysql_executable_comment_keyword", "mysql_comment", "MySQL executable comments preserve keyword execution while changing surface form.", op_mysql_executable_comment_keyword),
    SqlMutationOperator("many_mysql_executable_comments", "mysql_comment", "MySQL executable comments are applied to all recognized SQL keywords.", op_many_mysql_executable_comments),
]


OPERATOR_SETS: dict[str, list[SqlMutationOperator]] = {
    "conservative": CONSERVATIVE_OPERATORS,
    "wafamole_style": WAFAMOLE_STYLE_OPERATORS,
}


# ── AdvSQLi operators (Table I, Qu et al. 2024, arXiv:2401.02615) ─────────────
# Faithful to the paper.  Three semantic fixes applied vs. the paper's examples:
#   • \xa0 omitted from whitespace pool — MySQL does not treat it as whitespace.
#   • Comment Rewriting uses only alphanumeric replacement strings to prevent
#     accidental */ injection inside a comment body.
#   • Integer Encoding guards against (select n) inside LIMIT clauses, which
#     MySQL rejects as a syntax error.

# Whitespace characters the paper marks with * (request-method-flexible).
# The paper shows \t and \n; we add \r \v \f but exclude \xa0.
_ADVSQLI_WHITESPACES = [" ", "\t", "\n", "\r", "\v", "\f"]

# Fixed comment payloads for injection (paper examples: /*foo*/, /*bar*/).
_ADVSQLI_COMMENTS = ["/*foo*/", "/*bar*/", "/*baz*/", "/*1*/", "/*x*/"]

# Tautology / contradiction forms from the paper (Table I, Figure 2).
_ADVSQLI_TRUE_FORMS = [
    "2 <> 3",
    "TRUE",
    "NOT FALSE",
    "0x2 = 2",
    "rand() >= 0",
    "(select ord('r') regexp 114) = 0x1",
    "'foo' like 'foo'",
]
_ADVSQLI_FALSE_FORMS = [
    "FALSE",
    "0",
    "2 = 3",
    "'foo' like 'bar'",
]


def _advsqli_find_spans(pattern: str, text: str, flags: int = 0) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(pattern, text, flags)]


def _advsqli_replace(text: str, span: tuple[int, int, str], repl: str) -> str:
    s, e, _ = span
    return text[:s] + repl + text[e:]


def _advsqli_split_trailing_comment(text: str) -> tuple[str, str]:
    m = re.search(r"(--[^\n\r]*|#[^\n\r]*)", text)
    return (text[: m.start()], text[m.start():]) if m else (text, "")


# 1. Case Swapping — or 1=1 → oR 1=1
def op_advsqli_case_swapping(text: str, rng: random.Random) -> str:
    try:
        import sqlparse  # type: ignore
        parsed: list = []
        for stmt in sqlparse.parse(text):
            parsed.extend(stmt.flatten())
        kws = set(sqlparse.keywords.KEYWORDS_COMMON.keys())
        out, changed = [], False
        for tok in parsed:
            if tok.value.upper() in kws and rng.random() < 0.75:
                out.append(_randomize_case(tok.value, rng))
                changed = True
            else:
                out.append(tok.value)
        if changed:
            return "".join(out)
    except Exception:
        pass
    # Fallback: use the pre-compiled keyword regex.
    return _KEYWORD_RE.sub(lambda m: _randomize_case(m.group(0), rng), text)


# 2. Whitespace Substitution — or 1=1 → \tor1\n=1  (* flexible)
def op_advsqli_whitespace_substitution(text: str, rng: random.Random) -> str:
    matches = list(re.finditer(r"\s+", text))
    if not matches:
        return text
    m = rng.choice(matches)
    return text[: m.start()] + rng.choice(_ADVSQLI_WHITESPACES) + text[m.end():]


# 3. Comment Injection — or 1=1 → /*foo*/or 1=/*bar*/1  (* flexible)
def op_advsqli_comment_injection(text: str, rng: random.Random) -> str:
    spans = _advsqli_find_spans(r"\b(union|select|from|where|or|and)\b", text, re.I)
    if not spans:
        return text
    span = rng.choice(spans)
    comment = rng.choice(_ADVSQLI_COMMENTS)
    repl = (comment + span[2]) if rng.random() < 0.5 else (span[2] + comment)
    return _advsqli_replace(text, span, repl)


# 4. Comment Rewriting — /*foo*/or 1=1 → /*1.png*/or 1=1
def op_advsqli_comment_rewriting(text: str, rng: random.Random) -> str:
    multiline = list(re.finditer(r"/\*.*?\*/", text, re.S))
    if multiline:
        target = rng.choice(multiline)
        # Use only alphanumeric labels — no punctuation that could inject */.
        label = rng.choice(["1png", "bar", "hello", "img", "x1"])
        repl = f"/*{label}*/"
        return text[: target.start()] + repl + text[target.end():]
    single = list(re.finditer(r"(--[^\n\r]*|#[^\n\r]*)", text))
    if single:
        target = rng.choice(single)
        return text[: target.end()] + rng.choice(["abc", "1", "x"]) + text[target.end():]
    return text


# 5. Integer Encoding — or 1=1 → or 0x1=1
def op_advsqli_integer_encoding(text: str, rng: random.Random) -> str:
    mask = _quote_mask(text)
    candidates = [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(r"(?<![%A-Za-z0-9_x])\d+(?![A-Za-z0-9_])", text)
        if _outside_quotes(mask, m.start(), m.end())
    ]
    if not candidates:
        return text
    s, e, val_str = rng.choice(candidates)
    value = int(val_str)
    in_limit = bool(re.search(r"\bLIMIT\b[^;]*$", text[:s], re.IGNORECASE))
    replacements = [hex(value)]
    if not in_limit:
        replacements.append(f"(select {value})")
    return text[:s] + rng.choice(replacements) + text[e:]


# 6. Operator Swapping — or 1=1 → or 1 like 1
def op_advsqli_operator_swapping(text: str, rng: random.Random) -> str:
    candidates: list[tuple[int, int, str, list[str]]] = []
    candidates += [(s, e, g, ["||"]) for s, e, g in _advsqli_find_spans(r"\bOR\b", text, re.I)]
    candidates += [(s, e, g, ["&&"]) for s, e, g in _advsqli_find_spans(r"\bAND\b", text, re.I)]
    candidates += [(s, e, g, [" like "]) for s, e, g in _advsqli_find_spans(r"(?<![<>!])=(?!=)", text)]
    if not candidates:
        return text
    s, e, _, repls = rng.choice(candidates)
    return text[:s] + rng.choice(repls) + text[e:]


# 7. Logical Invariant — or 1=1 → or 1=1 and 'a'='a'
def op_advsqli_logical_invariant(text: str, rng: random.Random) -> str:
    body, tail = _advsqli_split_trailing_comment(text)
    matches: list[re.Match[str]] = []
    for pat in [r"\b\d+(?:\.\d+)?\s*(?:=|like)\s*\d+(?:\.\d+)?\b",
                r"'[^']+'\s*(?:=|like)\s*'[^']+'"]:
        matches.extend(re.finditer(pat, body, re.I))
    if not matches:
        return text
    target = rng.choice(matches)
    suffix = rng.choice([" AND 'a' = 'a'", " AND TRUE", " AND 1", " OR FALSE", " OR 0"])
    return body[: target.end()] + suffix + body[target.end():] + tail


# 8. Inline Comment — union select → /*!union*/ /*!50000select*/
def op_advsqli_inline_comment(text: str, rng: random.Random) -> str:
    if "/*!" in text:
        return text
    spans = []
    for kw in ["union", "select", "where", "from"]:
        spans.extend(_advsqli_find_spans(rf"\b{kw}\b", text, re.I))
    if not spans:
        return text
    span = rng.choice(spans)
    kw = span[2]
    repl = f"/*!50000{kw}*/" if kw.lower() == "select" else f"/*!{kw}*/"
    return _advsqli_replace(text, span, repl)


# 9. Where Rewriting — where xxx → where xxx and True  /  where (select 0) or xxx
def op_advsqli_where_rewriting(text: str, rng: random.Random) -> str:
    body, tail = _advsqli_split_trailing_comment(text)
    m = re.search(r"\bwhere\b", body, re.I)
    if not m:
        return text
    before = body[: m.start()]
    kw = body[m.start(): m.end()]
    after = body[m.end():].lstrip()
    if not after:
        return text
    if rng.random() < 0.5:
        return before + kw + " " + after + " AND TRUE" + tail
    return before + kw + " (select 0) OR " + after + tail


# 10. DML Substitution — or 1=1 → || 1=1  /  and name='foo' → && name='foo'  (* flexible)
def op_advsqli_dml_substitution(text: str, rng: random.Random) -> str:
    del rng
    if re.search(r"\bOR\b", text, re.I):
        return re.sub(r"\bOR\b", "||", text, count=1, flags=re.I)
    if re.search(r"\bAND\b", text, re.I):
        return re.sub(r"\bAND\b", "&&", text, count=1, flags=re.I)
    return text


# 11. Tautology Substitution — '1'='1' → 2<>3  /  1=1 → rand()>=0
def op_advsqli_tautology_substitution(text: str, rng: random.Random) -> str:
    candidates = []
    for pat in [r"\b\d+(?:\.\d+)?\s*(?:=|like)\s*\d+(?:\.\d+)?\b",
                r"'[^']+'\s*(?:=|like)\s*'[^']+'"]:
        candidates.extend(_advsqli_find_spans(pat, text, re.I))
    if not candidates:
        return text
    span = rng.choice(candidates)
    pool = _ADVSQLI_TRUE_FORMS if re.search(r"(=|like)", span[2], re.I) else _ADVSQLI_FALSE_FORMS
    repl = rng.choice(pool)
    if repl == span[2]:
        repl = rng.choice([x for x in pool if x != span[2]] or pool)
    return _advsqli_replace(text, span, repl)


ADVSQLI_OPERATORS = [
    SqlMutationOperator("advsqli_case_swapping",             "surface_obfuscation", "AdvSQLi: randomise case of SQL keywords.",                         op_advsqli_case_swapping),
    SqlMutationOperator("advsqli_whitespace_substitution",   "surface_obfuscation", "AdvSQLi: replace one whitespace run with an alternative character.", op_advsqli_whitespace_substitution),
    SqlMutationOperator("advsqli_comment_injection",         "surface_obfuscation", "AdvSQLi: inject a comment token before/after a core keyword.",       op_advsqli_comment_injection),
    SqlMutationOperator("advsqli_comment_rewriting",         "surface_obfuscation", "AdvSQLi: replace existing comment content with a benign string.",    op_advsqli_comment_rewriting),
    SqlMutationOperator("advsqli_integer_encoding",          "numeric_repr",        "AdvSQLi: encode an integer as hex or scalar SELECT.",                 op_advsqli_integer_encoding),
    SqlMutationOperator("advsqli_operator_swapping",         "operator_synonym",    "AdvSQLi: swap OR/AND/= with || / && / LIKE.",                         op_advsqli_operator_swapping),
    SqlMutationOperator("advsqli_logical_invariant",         "boolean_equivalent",  "AdvSQLi: append AND/OR invariant after a tautology.",                 op_advsqli_logical_invariant),
    SqlMutationOperator("advsqli_inline_comment",            "mysql_comment",       "AdvSQLi: wrap a keyword in MySQL executable comment /*!…*/.",          op_advsqli_inline_comment),
    SqlMutationOperator("advsqli_where_rewriting",           "boolean_equivalent",  "AdvSQLi: structurally rewrite the WHERE clause.",                     op_advsqli_where_rewriting),
    SqlMutationOperator("advsqli_dml_substitution",          "operator_synonym",    "AdvSQLi: substitute OR→|| or AND→&&.",                                op_advsqli_dml_substitution),
    SqlMutationOperator("advsqli_tautology_substitution",    "boolean_equivalent",  "AdvSQLi: replace a tautology with an equivalent form.",               op_advsqli_tautology_substitution),
]

OPERATOR_SETS["advsqli"] = ADVSQLI_OPERATORS


_LIMIT_SUBQUERY_RE = re.compile(r"\bLIMIT\b[^;]*\(SELECT\b", re.IGNORECASE)


def _is_official_output_valid(text: str) -> bool:
    """
    Guard against three known semantic bugs in official WAF-A-MoLE operators:

    1. spaces_to_whitespaces_alternatives injects \\xa0, which MySQL does NOT
       treat as whitespace — the query becomes syntactically invalid.

    2. comment_rewriting builds /*<random_string>*/ but random_string() draws
       from string.punctuation, so it can produce '*/' inside the comment,
       closing it early and exposing the remainder as raw SQL.

    3. swap_int_repr wraps integers as (SELECT n), but applying this inside a
       LIMIT clause (e.g. LIMIT (SELECT 10)) is illegal in MySQL.
    """
    # Bug 1: \xa0 is not a valid MySQL token separator.
    if "\xa0" in text:
        return False

    # Bug 2: walk the text tracking comment depth; a negative depth means a
    # spurious */ closed a comment that was never opened (or a random_string
    # injected */ inside an existing comment, splitting it).
    depth = 0
    i = 0
    while i < len(text):
        if text[i : i + 2] == "/*":
            depth += 1
            i += 2
        elif text[i : i + 2] == "*/":
            depth -= 1
            if depth < 0:
                return False  # */ outside any open comment
            i += 2
        else:
            i += 1
    if depth != 0:
        return False  # unclosed /*

    # Bug 3: (SELECT …) is not a valid argument to LIMIT in MySQL.
    if _LIMIT_SUBQUERY_RE.search(text):
        return False

    return True


def _official_wafamole_operator_set() -> list[SqlMutationOperator]:
    wafamole_root = Path(__file__).resolve().parents[1] / "external" / "WAF-A-MoLE"
    if str(wafamole_root) not in sys.path:
        sys.path.insert(0, str(wafamole_root))

    try:
        from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard.
        raise RuntimeError(f"could not import official WAF-A-MoLE SqlFuzzer from {wafamole_root}") from exc

    operators: list[SqlMutationOperator] = []
    for strategy in SqlFuzzer.strategies:

        def make_fn(fn: Callable[[str], str]) -> MutationFn:
            def wrapped(text: str, rng: random.Random) -> str:
                result = _call_with_rng(fn, text, rng)
                # Treat structurally broken outputs as no-ops so _candidate_texts
                # discards them rather than scoring invalid SQL against the model.
                return result if _is_official_output_valid(result) else text

            return wrapped

        operators.append(
            SqlMutationOperator(
                name=f"official_{strategy.__name__}",
                family="official_wafamole",
                semantic_note="Official WAF-A-MoLE SQL fuzzer strategy.",
                fn=make_fn(strategy),
            )
        )
    return operators


def get_operator_set(name: str) -> list[SqlMutationOperator]:
    if name == "official_wafamole":
        return _official_wafamole_operator_set()
    if name not in OPERATOR_SETS:
        raise KeyError(f"unknown operator set: {name}")
    return OPERATOR_SETS[name]


def random_operator_chain(
    source_text: str,
    seed: int,
    operators: Iterable[SqlMutationOperator],
    rounds: int = 3,
    retries: int = 8,
    max_chars: int = 640,
    ensure_changed: bool = True,
) -> RandomMutationResult:
    """Build a non-targeted mutation chain for training augmentation."""

    operator_list = list(operators)
    if not operator_list:
        raise ValueError("operators must not be empty")

    best_text = source_text
    best_chain: tuple[str, ...] = ()
    for attempt in range(max(1, retries)):
        rng = random.Random(seed + attempt * 104_729)
        current = source_text
        chain: list[str] = []
        for _ in range(max(1, rounds)):
            op = rng.choice(operator_list)
            mutated = op.fn(current, rng)
            if mutated == current or len(mutated) > max_chars:
                continue
            current = mutated
            chain.append(op.name)

        if current != source_text:
            return RandomMutationResult(
                source_text=source_text,
                mutated_text=current,
                changed=True,
                chain=tuple(chain),
            )
        if not ensure_changed:
            return RandomMutationResult(
                source_text=source_text,
                mutated_text=current,
                changed=False,
                chain=tuple(chain),
            )
        if len(chain) > len(best_chain):
            best_text = current
            best_chain = tuple(chain)

    return RandomMutationResult(
        source_text=source_text,
        mutated_text=best_text,
        changed=best_text != source_text,
        chain=best_chain,
    )


def _candidate_texts(
    state: CandidateState,
    operators: list[SqlMutationOperator],
    rng: random.Random,
    candidates_per_state: int,
    max_chars: int,
    grad_fn: "GradFn | None" = None,
) -> list[tuple[str, tuple[str, ...]]]:
    # Compute gradient-guided operator weights once per state (one forward+backward pass).
    op_weights: list[float] | None = None
    if grad_fn is not None:
        try:
            importances = grad_fn(state.text)
            if importances is not None and len(importances) > 0:
                from experiments.tokenization import tokenize_sql  # local import to avoid circular
                tokens = tokenize_sql(state.text)
                op_weights = _operator_weights_from_importances(operators, tokens, importances)
        except Exception:
            op_weights = None

    candidates: list[tuple[str, tuple[str, ...]]] = []
    seen = {state.text}
    for _ in range(candidates_per_state):
        if op_weights is not None:
            (op,) = rng.choices(operators, weights=op_weights, k=1)
        else:
            op = rng.choice(operators)
        mutated = op.fn(state.text, rng)
        if mutated == state.text or mutated in seen:
            continue
        if len(mutated) > max_chars:
            continue
        seen.add(mutated)
        candidates.append((mutated, state.chain + (op.name,)))
    return candidates


def targeted_evasion_search(
    source_text: str,
    score_fn: ScoreFn,
    seed: int,
    operators: Iterable[SqlMutationOperator],
    steps: int = 12,
    candidates_per_state: int = 24,
    beam_size: int = 3,
    success_threshold: float = 0.5,
    max_chars: int = 640,
    early_stop: bool = True,
    grad_fn: "GradFn | None" = None,
) -> TargetedSearchResult:
    rng = random.Random(seed)
    operator_list = list(operators)
    if not operator_list:
        raise ValueError("operators must not be empty")

    score_cache: dict[str, float] = {}

    def score_texts(texts: list[str]) -> list[float]:
        missing = list(dict.fromkeys(text for text in texts if text not in score_cache))
        if missing:
            probs = [float(x) for x in score_fn(missing)]
            score_cache.update(zip(missing, probs))
        return [score_cache[text] for text in texts]

    source_prob = score_texts([source_text])[0]
    best = CandidateState(text=source_text, prob=source_prob, chain=())
    if early_stop and source_prob < success_threshold:
        return TargetedSearchResult(
            source_text=source_text,
            adversarial_text=source_text,
            source_prob=source_prob,
            adversarial_prob=source_prob,
            success=True,
            changed=False,
            steps=0,
            queries=1,
            chain=(),
            history=(),
        )

    beam = [best]
    history: list[dict] = []
    visited_texts = {source_text}
    queries = 1

    for step in range(1, steps + 1):
        raw_candidates: list[tuple[str, tuple[str, ...]]] = []
        seen_texts = {state.text for state in beam}
        for state in beam:
            for text, chain in _candidate_texts(state, operator_list, rng, candidates_per_state, max_chars, grad_fn=grad_fn):
                if text in seen_texts or text in visited_texts:
                    continue
                seen_texts.add(text)
                raw_candidates.append((text, chain))

        if not raw_candidates:
            history.append({"step": step, "candidates": 0, "best_prob": best.prob})
            break

        texts = [item[0] for item in raw_candidates]
        probs = score_texts(texts)
        visited_texts.update(texts)
        queries += len(texts)
        ranked = sorted(zip(raw_candidates, probs), key=lambda item: item[1])

        step_states = [
            CandidateState(text=text, prob=prob, chain=chain)
            for (text, chain), prob in ranked[: max(beam_size, 1)]
        ]
        if step_states and step_states[0].prob < best.prob:
            best = step_states[0]
        beam = sorted(step_states + beam, key=lambda item: item.prob)[: max(beam_size, 1)]
        history.append(
            {
                "step": step,
                "candidates": len(texts),
                "best_prob": best.prob,
                "best_chain": list(best.chain),
            }
        )
        if early_stop and best.prob < success_threshold:
            break

    return TargetedSearchResult(
        source_text=source_text,
        adversarial_text=best.text,
        source_prob=source_prob,
        adversarial_prob=best.prob,
        success=best.prob < success_threshold,
        changed=best.text != source_text,
        steps=len(history),
        queries=queries,
        chain=best.chain,
        history=tuple(history),
    )


def targeted_evasion_search_many(
    source_texts: list[str],
    score_fn: ScoreFn,
    seeds: list[int],
    operators: Iterable[SqlMutationOperator],
    steps: int = 12,
    candidates_per_state: int = 24,
    beam_size: int = 3,
    success_threshold: float = 0.5,
    max_chars: int = 640,
    early_stop: bool = True,
    grad_fn: "GradFn | None" = None,
) -> list[TargetedSearchResult]:
    if len(source_texts) != len(seeds):
        raise ValueError("source_texts and seeds must have the same length")

    operator_list = list(operators)
    if not operator_list:
        raise ValueError("operators must not be empty")

    global_score_cache: dict[str, float] = {}

    def score_texts(texts: list[str]) -> list[float]:
        missing = list(dict.fromkeys(text for text in texts if text not in global_score_cache))
        if missing:
            probs = [float(x) for x in score_fn(missing)]
            global_score_cache.update(zip(missing, probs))
        return [global_score_cache[text] for text in texts]

    source_probs = score_texts(source_texts)
    sessions: list[_SearchSession] = []
    for source_text, seed, source_prob in zip(source_texts, seeds, source_probs):
        best = CandidateState(text=source_text, prob=source_prob, chain=())
        done = bool(early_stop and source_prob < success_threshold)
        sessions.append(
            _SearchSession(
                source_text=source_text,
                source_prob=source_prob,
                rng=random.Random(seed),
                best=best,
                beam=[best],
                visited_texts={source_text},
                queries=1,
                history=[],
                done=done,
            )
        )

    for step in range(1, steps + 1):
        pending: list[tuple[_SearchSession, list[tuple[str, tuple[str, ...]]]]] = []
        batch_missing: list[str] = []
        batch_missing_seen: set[str] = set()
        any_active = False

        for session in sessions:
            if session.done:
                continue
            any_active = True
            raw_candidates: list[tuple[str, tuple[str, ...]]] = []
            seen_texts = {state.text for state in session.beam}
            for state in session.beam:
                for text, chain in _candidate_texts(
                    state,
                    operator_list,
                    session.rng,
                    candidates_per_state,
                    max_chars,
                    grad_fn=grad_fn,
                ):
                    if text in seen_texts or text in session.visited_texts:
                        continue
                    seen_texts.add(text)
                    raw_candidates.append((text, chain))

            if not raw_candidates:
                session.history.append({"step": step, "candidates": 0, "best_prob": session.best.prob})
                session.done = True
                continue

            pending.append((session, raw_candidates))
            for text, _ in raw_candidates:
                if text in global_score_cache or text in batch_missing_seen:
                    continue
                batch_missing_seen.add(text)
                batch_missing.append(text)

        if not any_active:
            break

        if batch_missing:
            score_texts(batch_missing)

        for session, raw_candidates in pending:
            texts = [item[0] for item in raw_candidates]
            probs = [global_score_cache[text] for text in texts]
            session.visited_texts.update(texts)
            session.queries += len(texts)
            ranked = sorted(zip(raw_candidates, probs), key=lambda item: item[1])

            step_states = [
                CandidateState(text=text, prob=prob, chain=chain)
                for (text, chain), prob in ranked[: max(beam_size, 1)]
            ]
            if step_states and step_states[0].prob < session.best.prob:
                session.best = step_states[0]
            session.beam = sorted(step_states + session.beam, key=lambda item: item.prob)[: max(beam_size, 1)]
            session.history.append(
                {
                    "step": step,
                    "candidates": len(texts),
                    "best_prob": session.best.prob,
                    "best_chain": list(session.best.chain),
                }
            )
            if early_stop and session.best.prob < success_threshold:
                session.done = True

    return [
        TargetedSearchResult(
            source_text=session.source_text,
            adversarial_text=session.best.text,
            source_prob=session.source_prob,
            adversarial_prob=session.best.prob,
            success=session.best.prob < success_threshold,
            changed=session.best.text != session.source_text,
            steps=len(session.history),
            queries=session.queries,
            chain=session.best.chain,
            history=tuple(session.history),
        )
        for session in sessions
    ]
