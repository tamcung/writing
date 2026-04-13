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


SQL_KEYWORDS = [
    "select",
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


@dataclass(frozen=True)
class RandomMutationResult:
    source_text: str
    mutated_text: str
    changed: bool
    chain: tuple[str, ...]


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
    keyword_re = r"\b(" + "|".join(re.escape(k) for k in SQL_KEYWORDS) + r")\b"

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        mutated = _randomize_case(token, rng)
        return mutated if mutated != token else token.swapcase()

    return re.sub(keyword_re, repl, text, flags=re.IGNORECASE)


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
    keyword_re = r"\b(" + "|".join(re.escape(k) for k in SQL_KEYWORDS) + r")\b"

    def repl(match: re.Match[str]) -> str:
        token = match.group(0).upper()
        return f"/*!50000{token}*/"

    return _replace_random_match(text, keyword_re, repl, rng, flags=re.IGNORECASE)


def op_many_mysql_executable_comments(text: str, rng: random.Random) -> str:
    del rng
    keyword_re = r"\b(" + "|".join(re.escape(k) for k in SQL_KEYWORDS) + r")\b"

    def repl(match: re.Match[str]) -> str:
        return f"/*!50000{match.group(0).upper()}*/"

    return re.sub(keyword_re, repl, text, flags=re.IGNORECASE)


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


def _official_wafamole_operator_set() -> list[SqlMutationOperator]:
    wafamole_root = Path(__file__).resolve().parents[2] / "external" / "WAF-A-MoLE"
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
                state = random.getstate()
                random.setstate(rng.getstate())
                try:
                    mutated = fn(text)
                    rng.setstate(random.getstate())
                    return mutated
                finally:
                    random.setstate(state)

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
) -> list[tuple[str, tuple[str, ...]]]:
    candidates: list[tuple[str, tuple[str, ...]]] = []
    seen = {state.text}
    for _ in range(candidates_per_state):
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
            for text, chain in _candidate_texts(state, operator_list, rng, candidates_per_state, max_chars):
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
