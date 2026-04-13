#!/usr/bin/env python3
"""Compare augmentation-only vs representation consistency for SQLi payloads.

The experiment keeps the augmented training examples identical for the augmented
baselines. The only difference between `aug_ce` and `aug_consistency` is the
extra representation-level consistency loss between each malicious payload and
its semantics-preserving mutation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import re
import sys
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlparse
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_URL = "https://raw.githubusercontent.com/nidnogg/sqliv5-dataset/main/SQLiV3_clean.json"
SQL_KEYWORDS = [
    "select",
    "union",
    "where",
    "and",
    "or",
    "like",
    "sleep",
    "benchmark",
    "from",
    "information_schema",
    "concat",
    "char",
    "order",
    "by",
    "group",
    "having",
    "insert",
    "update",
    "delete",
    "drop",
    "create",
    "alter",
    "table",
    "user",
    "password",
    "null",
    "true",
    "false",
]
FULL_SQL_RE = re.compile(r"^\s*(select|insert|update|delete|create|drop|alter|truncate|merge|replace)\b", re.I)
TOKEN_SPLIT_RE = re.compile(r"(\s+|/\*.*?\*/|--[^\n\r]*)", re.S)
_OFFICIAL_SQLFUZZER = None
ADVSQLI_TRUE_FORMS = [
    "2 <> 3",
    "TRUE",
    "NOT FALSE",
    "0x2 = 2",
    "(select ord('r') regexp 114) = 0x1",
    "'foo' like 'foo'",
]
ADVSQLI_FALSE_FORMS = [
    "0",
    "FALSE",
    "select 0",
    "2 = 3",
    "'foo' like 'bar'",
]
ADVSQLI_WHITESPACES = [" ", "\t", "\n", "\r", "\v", "\f", "\xa0"]
ADVSQLI_COMMENTS = ["/*foo*/", "/*bar*/", "/*1.png*/", "/*a21r*!*/", "/*hello world*/"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def ensure_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {DATA_URL} -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_payload_data(path: Path, max_len: int) -> tuple[list[str], list[int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_text: dict[str, set[int]] = {}

    for row in rows:
        text = str(row.get("pattern", "")).strip()
        typ = str(row.get("type", "")).strip().lower()
        if not text or len(text) > max_len:
            continue
        if typ not in {"sqli", "valid"}:
            continue

        label = 1 if typ == "sqli" else 0
        if label == 1 and FULL_SQL_RE.search(text):
            continue

        by_text.setdefault(text, set()).add(label)

    texts: list[str] = []
    labels: list[int] = []
    for text, label_set in by_text.items():
        if len(label_set) != 1:
            continue
        labels.append(next(iter(label_set)))
        texts.append(text)

    return texts, labels


def rand_case(token: str, rng: random.Random) -> str:
    return "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in token)


def mutate_keywords(text: str, rng: random.Random) -> str:
    out = text
    for kw in SQL_KEYWORDS:
        pat = re.compile(rf"\b{re.escape(kw)}\b", re.I)
        out = pat.sub(lambda m: rand_case(m.group(0), rng), out)
    return out


def mutate_tautologies(text: str, rng: random.Random) -> str:
    true_forms = ["0x1 LIKE 0x1", "'a'='a'", "2>1", "TRUE", "1 IN (1)"]
    false_forms = ["0x1 LIKE 0x2", "'a'<>'a'", "2<1", "FALSE", "1 IN (2)"]
    out = re.sub(r"\b1\s*=\s*1\b", lambda _: rng.choice(true_forms), text)
    out = re.sub(r"\b1\s*=\s*2\b", lambda _: rng.choice(false_forms), out)
    out = re.sub(r"\b2\s*=\s*2\b", lambda _: rng.choice(true_forms), out)
    return out


def mutate_separators(text: str, rng: random.Random, hard: bool) -> str:
    separators = [" ", "/**/", "\t", "\n", "\x0b", "\x0c"]
    hard_separators = [" ", "/**/", "/*x*/", "\t", "\n", "\xa0", "/**/\n/**/"]

    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        if not value.strip():
            return rng.choice(hard_separators if hard else separators)
        return value

    return TOKEN_SPLIT_RE.sub(repl, text)


def mutate_operators(text: str, rng: random.Random) -> str:
    out = re.sub(r"\bOR\b", lambda _: rng.choice(["OR", "||", "or"]), text, flags=re.I)
    out = re.sub(r"\bAND\b", lambda _: rng.choice(["AND", "&&", "and"]), out, flags=re.I)
    return out


def append_comment_noise(text: str, rng: random.Random) -> str:
    if "#" in text or "--" in text:
        return text + rng.choice(["", "x", "7", "abc"])
    if rng.random() < 0.35:
        return text + rng.choice([" --", "#", "/*tail*/"])
    return text


def _find_spans(pattern: str, text: str, flags: int = 0) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(pattern, text, flags)]


def _replace_span(text: str, span: tuple[int, int, str], repl: str) -> str:
    start, end, _ = span
    return text[:start] + repl + text[end:]


def _split_trailing_line_comment(text: str) -> tuple[str, str]:
    match = re.search(r"(--[^\n\r]*|#[^\n\r]*)", text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.start() :]


def advsqli_case_swapping(text: str, rng: random.Random) -> str:
    parsed = []
    try:
        for stmt in sqlparse.parse(text):
            parsed.extend(list(stmt.flatten()))
    except Exception:
        return mutate_keywords(text, rng)
    if not parsed:
        return mutate_keywords(text, rng)
    sql_keywords = set(sqlparse.keywords.KEYWORDS_COMMON.keys())
    out = []
    changed = False
    for token in parsed:
        value = token.value
        if value.upper() in sql_keywords and rng.random() < 0.75:
            out.append(rand_case(value, rng))
            changed = True
        else:
            out.append(value)
    return "".join(out) if changed else mutate_keywords(text, rng)


def advsqli_whitespace_substitution(text: str, rng: random.Random, hard: bool) -> str:
    matches = list(re.finditer(r"\s+", text))
    if not matches:
        return text
    match = rng.choice(matches)
    replacement = rng.choice(ADVSQLI_WHITESPACES if not hard else ADVSQLI_WHITESPACES + ["/**/", "/*foo*/"])
    return text[: match.start()] + replacement + text[match.end() :]


def advsqli_comment_injection(text: str, rng: random.Random) -> str:
    candidates = _find_spans(r"\b(union|select|from|where)\b", text, flags=re.I)
    if not candidates:
        return text
    span = rng.choice(candidates)
    if "/*" in span[2] or "*/" in span[2]:
        return text
    if rng.random() < 0.5:
        repl = f"{rng.choice(ADVSQLI_COMMENTS)}{span[2]}"
    else:
        repl = f"{span[2]}{rng.choice(ADVSQLI_COMMENTS)}"
    return _replace_span(text, span, repl)


def advsqli_comment_rewriting(text: str, rng: random.Random) -> str:
    multiline = list(re.finditer(r"/\*.*?\*/", text))
    if multiline:
        target = rng.choice(multiline)
        repl = f"/*{rng.choice(['1.png', 'a21r*!', 'hello world', 'provincia=burgos'])}*/"
        return text[: target.start()] + repl + text[target.end() :]
    single = list(re.finditer(r"(--[^\n\r]*|#[^\n\r]*)", text))
    if single:
        target = rng.choice(single)
        return text[: target.end()] + rng.choice(["abc", "1.gif", "login=karina9"]) + text[target.end() :]
    return text


def advsqli_integer_encoding(text: str, rng: random.Random) -> str:
    candidates = []
    for span in _find_spans(r"\b\d+\b", text):
        start, end, _ = span
        left = text[start - 1] if start > 0 else ""
        right = text[end] if end < len(text) else ""
        if left in {"'", '"'} or right in {"'", '"'}:
            continue
        candidates.append(span)
    if not candidates:
        return text
    span = rng.choice(candidates)
    value = int(span[2])
    repl = rng.choice([hex(value), f"(select {value})"])
    return _replace_span(text, span, repl)


def advsqli_operator_swapping(text: str, rng: random.Random) -> str:
    candidates = []
    candidates.extend((s, e, g, ["||"]) for s, e, g in _find_spans(r"\bOR\b", text, flags=re.I))
    candidates.extend((s, e, g, ["&&"]) for s, e, g in _find_spans(r"\bAND\b", text, flags=re.I))
    candidates.extend((s, e, g, [" like "]) for s, e, g in _find_spans(r"=", text))
    if not candidates:
        return text
    start, end, _, repls = rng.choice(candidates)
    return text[:start] + rng.choice(repls) + text[end:]


def advsqli_logical_invariant(text: str, rng: random.Random) -> str:
    body, tail = _split_trailing_line_comment(text)
    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:=|like)\s*\d+(?:\.\d+)?\b",
        r"'[^']+'\s*(?:=|like)\s*'[^']+'",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(list(re.finditer(pattern, body, re.I)))
    if not matches:
        return text
    target = rng.choice(matches)
    invariant = rng.choice(
        [
            " AND 'a' = 'a'",
            " AND TRUE",
            " AND 1",
            " OR FALSE",
            " OR 0",
        ]
    )
    return body[: target.end()] + invariant + body[target.end() :] + tail


def advsqli_inline_comment(text: str, rng: random.Random) -> str:
    if "/*!" in text:
        return text
    candidates = []
    for kw in ["union", "select", "where", "from"]:
        candidates.extend(_find_spans(rf"\b{kw}\b", text, flags=re.I))
    if not candidates:
        return text
    span = rng.choice(candidates)
    kw = span[2]
    if kw.lower() == "select":
        repl = f"/*!50000{kw}*/"
    else:
        repl = f"/*!{kw}*/"
    return _replace_span(text, span, repl)


def advsqli_where_rewriting(text: str, rng: random.Random) -> str:
    body, tail = _split_trailing_line_comment(text)
    match = re.search(r"\bwhere\b", body, re.I)
    if not match:
        return text
    after = body[match.end() :]
    before = body[: match.start()]
    stripped_after = after.lstrip()
    if not stripped_after:
        return text
    if rng.random() < 0.5:
        return before + body[match.start() : match.end()] + " " + stripped_after + " AND TRUE" + tail
    return before + body[match.start() : match.end()] + " (select 0) OR " + stripped_after + tail


def advsqli_dml_substitution(text: str, rng: random.Random) -> str:
    if re.search(r"\bOR\b", text, re.I):
        return re.sub(r"\bOR\b", "||", text, count=1, flags=re.I)
    if re.search(r"\bAND\b", text, re.I):
        return re.sub(r"\bAND\b", "&&", text, count=1, flags=re.I)
    return text


def advsqli_tautology_substitution(text: str, rng: random.Random) -> str:
    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:=|like)\s*\d+(?:\.\d+)?\b",
        r"'[^']+'\s*(?:=|like)\s*'[^']+'",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(_find_spans(pattern, text, flags=re.I))
    if not candidates:
        return mutate_tautologies(text, rng)
    span = rng.choice(candidates)
    current = span[2]
    replacement_pool = ADVSQLI_TRUE_FORMS if re.search(r"(=|like)", current, re.I) else ADVSQLI_FALSE_FORMS
    replacement = rng.choice(replacement_pool)
    if replacement == current:
        replacement = rng.choice([x for x in replacement_pool if x != current] or replacement_pool)
    return _replace_span(text, span, replacement)


def advsqli_style_mutation(text: str, seed: int, hard: bool = False, rounds: int = 1) -> str:
    """Approximate the transformation layer of AdvSQLi.

    This reproduces the paper's mutation families (tree/CFG-inspired) but not
    the full hierarchical-tree + MCTS search procedure.
    """
    rng = random.Random(seed)
    out = text
    semantic_ops = [
        (advsqli_case_swapping, 1.0),
        (lambda s, r: advsqli_whitespace_substitution(s, r, hard=hard), 1.2),
        (advsqli_integer_encoding, 1.0),
        (advsqli_operator_swapping, 1.0),
        (advsqli_logical_invariant, 0.8),
        (advsqli_where_rewriting, 0.6),
        (advsqli_dml_substitution, 0.9),
        (advsqli_tautology_substitution, 1.2),
    ]
    comment_ops = [
        (advsqli_comment_injection, 1.0),
        (advsqli_comment_rewriting, 0.6),
        (advsqli_inline_comment, 0.9),
    ]
    total_rounds = max(1, rounds * (2 if hard else 1))
    comment_applied = False
    for _ in range(total_rounds):
        semantic_weights = [w for _, w in semantic_ops]
        semantic_idx = rng.choices(range(len(semantic_ops)), weights=semantic_weights, k=1)[0]
        candidate = semantic_ops[semantic_idx][0](out, rng)
        if candidate != out:
            out = candidate
        if (not comment_applied) and rng.random() < (0.55 if hard else 0.25):
            comment_weights = [w for _, w in comment_ops]
            comment_idx = rng.choices(range(len(comment_ops)), weights=comment_weights, k=1)[0]
            candidate = comment_ops[comment_idx][0](out, rng)
            if candidate != out:
                out = candidate
                comment_applied = True
    return out[:260]


def semantic_mutation(text: str, seed: int, hard: bool = False, rounds: int = 1) -> str:
    rng = random.Random(seed)
    out = text
    ops = [mutate_keywords, mutate_tautologies, mutate_operators]
    for _ in range(rounds):
        rng.shuffle(ops)
        for op in ops:
            if rng.random() < (0.75 if hard else 0.55):
                out = op(out, rng)
        if rng.random() < (0.85 if hard else 0.55):
            out = mutate_separators(out, rng, hard=hard)
        if hard and rng.random() < 0.45:
            out = append_comment_noise(out, rng)
    return out[:260]


def get_official_sqlfuzzer(wafamole_repo: str):
    global _OFFICIAL_SQLFUZZER
    if _OFFICIAL_SQLFUZZER is None:
        repo = Path(wafamole_repo)
        if not repo.exists():
            raise FileNotFoundError(f"WAF-A-MoLE repo not found: {repo}")
        repo_str = str(repo.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        _OFFICIAL_SQLFUZZER = importlib.import_module("wafamole.payloadfuzzer.sqlfuzzer").SqlFuzzer
    return _OFFICIAL_SQLFUZZER


def official_wafamole_mutation(text: str, seed: int, hard: bool, rounds: int, wafamole_repo: str) -> str:
    """Apply the official WAF-A-MoLE SqlFuzzer strategies deterministically."""
    sql_fuzzer = get_official_sqlfuzzer(wafamole_repo)
    old_state = random.getstate()
    try:
        random.seed(seed)
        fuzzer = sql_fuzzer(text)
        out = text
        # Official SqlFuzzer applies one randomly chosen strategy per fuzz() call.
        # Hard mode gets more opportunities because some official strategies may no-op.
        total_rounds = rounds if not hard else rounds * 2
        for _ in range(max(1, total_rounds)):
            out = fuzzer.fuzz()
        return out[:260]
    finally:
        random.setstate(old_state)


def mutate_payload(
    text: str,
    seed: int,
    hard: bool,
    rounds: int,
    mutation_source: str,
    wafamole_repo: str,
) -> str:
    if mutation_source == "local":
        return semantic_mutation(text, seed=seed, hard=hard, rounds=rounds)
    if mutation_source == "advsqli":
        return advsqli_style_mutation(text, seed=seed, hard=hard, rounds=rounds)
    if mutation_source == "wafamole":
        return official_wafamole_mutation(
            text,
            seed=seed,
            hard=hard,
            rounds=rounds,
            wafamole_repo=wafamole_repo,
        )
    raise ValueError(f"Unknown mutation source: {mutation_source}")


def mutate_payload_with_retries(
    text: str,
    seed: int,
    hard: bool,
    rounds: int,
    mutation_source: str,
    wafamole_repo: str,
    ensure_changed: bool,
    retries: int,
) -> str:
    """Generate one mutated view, optionally retrying no-op mutations."""
    for attempt in range(max(1, retries)):
        out = mutate_payload(
            text,
            seed=seed + attempt * 104_729,
            hard=hard,
            rounds=rounds,
            mutation_source=mutation_source,
            wafamole_repo=wafamole_repo,
        )
        if not ensure_changed or out != text:
            return out
    return out


def make_split(
    texts: list[str],
    labels: list[int],
    seed: int,
    train_per_class: int,
    test_per_class: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    need = train_per_class + test_per_class
    if len(idx0) < need or len(idx1) < need:
        raise ValueError(f"Not enough data: benign={len(idx0)}, sqli={len(idx1)}, need={need}")

    rng = random.Random(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    train_idx = idx0[:train_per_class] + idx1[:train_per_class]
    test_idx = idx0[train_per_class:need] + idx1[train_per_class:need]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return (
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in test_idx],
    )


def build_vocab(texts: list[str]) -> dict[str, int]:
    chars = sorted({ch for text in texts for ch in text})
    return {ch: i + 2 for i, ch in enumerate(chars)}


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(ch, 1) for ch in text[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


class MultiViewDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        aug_texts: list[list[str]],
        vocab: dict[str, int],
        max_len: int,
        use_aug: bool,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.aug_texts = aug_texts
        self.vocab = vocab
        self.max_len = max_len
        self.use_aug = use_aug

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        y = self.labels[idx]
        views = [text]
        if self.use_aug and y == 1:
            views.extend(self.aug_texts[idx])
        else:
            views.extend([text] * len(self.aug_texts[idx]))
        x = torch.tensor([encode(view, self.vocab, self.max_len) for view in views], dtype=torch.long)
        return x, torch.tensor(y, dtype=torch.float32)


class EvalDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(encode(self.texts[idx], self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        channels: int = 48,
        dropout: float = 0.25,
        projected_classifier: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, channels, kernel_size=k, padding=k // 2) for k in (3, 5, 7)]
        )
        self.hidden_dim = channels * len(self.convs)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, 1)
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.canonical_classifier = nn.Linear(self.hidden_dim, 1)
        self.projected_classifier = projected_classifier

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            pooled.append(torch.max(h, dim=-1).values)
        return self.dropout(torch.cat(pooled, dim=1))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.projector(h)

    def classify(self, h: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if self.projected_classifier:
            if z is None:
                z = self.project(h)
            return self.canonical_classifier(z).squeeze(1)
        return self.classifier(h).squeeze(1)

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = self.project(h)
        logits = self.classify(h, z)
        rep = z if self.projected_classifier else h
        return logits, rep, h, z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, rep, _, _ = self.embed(x)
        return logits, rep


@dataclass
class TrainConfig:
    method: str
    seed: int
    epochs: int
    batch_size: int
    max_len: int
    lr: float
    consistency_weight: float
    device: str
    canonical_logit_weight: float = 0.25


def train_model(
    cfg: TrainConfig,
    train_texts: list[str],
    train_labels: list[int],
    train_aug_texts: list[list[str]],
    vocab: dict[str, int],
) -> CharCNN:
    set_seed(cfg.seed)
    projected_classifier = cfg.method in {"aug_proj_ce", "aug_canonical"}
    model = CharCNN(
        vocab_size=max(vocab.values(), default=1) + 1,
        projected_classifier=projected_classifier,
    ).to(cfg.device)
    use_aug = cfg.method in {"aug_ce", "aug_consistency", "aug_infonce", "aug_proj_ce", "aug_canonical"}
    dataset = MultiViewDataset(train_texts, train_labels, train_aug_texts, vocab, cfg.max_len, use_aug=use_aug)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(cfg.epochs):
        for x_views, y in loader:
            x_views = x_views.to(cfg.device)
            y = y.to(cfg.device)
            opt.zero_grad(set_to_none=True)

            batch_size, num_views, seq_len = x_views.shape
            logits_flat, rep_flat, h_flat, z_flat = model.embed(x_views.reshape(batch_size * num_views, seq_len))
            logits = logits_flat.view(batch_size, num_views)
            rep = rep_flat.view(batch_size, num_views, -1)
            h = h_flat.view(batch_size, num_views, -1)
            z = z_flat.view(batch_size, num_views, -1)
            y_expand = y.unsqueeze(1).expand(batch_size, num_views)
            ce = bce(logits.reshape(-1), y_expand.reshape(-1))

            loss = ce
            if cfg.method == "aug_consistency":
                mask = y > 0.5
                if mask.any() and num_views > 1:
                    anchor = rep[mask, 0].unsqueeze(1).expand(-1, num_views - 1, -1)
                    others = rep[mask, 1:]
                    sim = torch.cosine_similarity(anchor, others, dim=2)
                    loss = loss + cfg.consistency_weight * (1.0 - sim).mean()
            elif cfg.method == "aug_infonce":
                mask = y > 0.5
                if mask.any() and num_views > 1:
                    pos_views = rep[mask]
                    if pos_views.shape[1] == 2 and int(mask.sum().item()) > 1:
                        loss = loss + cfg.consistency_weight * pair_infonce_loss(
                            pos_views[:, 0, :], pos_views[:, 1, :]
                        )
                    else:
                        loss = loss + cfg.consistency_weight * multi_positive_supcon_loss(pos_views)
            elif cfg.method == "aug_canonical":
                mask = y > 0.5
                if mask.any() and num_views > 1:
                    canonical = z[mask, 0].detach().unsqueeze(1).expand(-1, num_views - 1, -1)
                    others = z[mask, 1:]
                    sim = torch.cosine_similarity(canonical, others, dim=2)
                    canonical_logits = logits[mask, 0].detach().unsqueeze(1).expand(-1, num_views - 1)
                    other_logits = logits[mask, 1:]
                    loss = loss + cfg.consistency_weight * (1.0 - sim).mean()
                    loss = loss + cfg.canonical_logit_weight * F.mse_loss(other_logits, canonical_logits)

            loss.backward()
            opt.step()

    return model


def pair_infonce_loss(h1: torch.Tensor, h2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    n = h1.shape[0]
    z = F.normalize(torch.cat([h1, h2], dim=0), dim=1)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=logits.device), -1e9)
    targets = torch.arange(2 * n, device=logits.device)
    targets = torch.where(targets < n, targets + n, targets - n)
    return F.cross_entropy(logits, targets)


def multi_positive_supcon_loss(h: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Supervised contrastive loss with multiple positive views per sample.

    h shape: [num_samples, num_views, dim]
    """
    num_samples, num_views, _ = h.shape
    z = F.normalize(h.reshape(num_samples * num_views, -1), dim=1)
    sample_ids = torch.arange(num_samples, device=h.device).repeat_interleave(num_views)
    logits = torch.matmul(z, z.T) / temperature
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(eye, -1e9)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    pos_mask = (sample_ids[:, None] == sample_ids[None, :]) & (~eye)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos.mean()


@torch.no_grad()
def predict_proba(
    model: CharCNN,
    texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
    max_len: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(EvalDataset(texts, labels, vocab, max_len), batch_size=batch_size, shuffle=False)
    probs = []
    ys = []
    for x, y in loader:
        logits, _ = model(x.to(device))
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
    return np.asarray(probs), np.asarray(ys, dtype=int)


@torch.no_grad()
def score_texts(
    model: CharCNN,
    texts: list[str],
    vocab: dict[str, int],
    max_len: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.asarray([], dtype=float)
    model.eval()
    probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        x = torch.tensor([encode(t, vocab, max_len) for t in batch], dtype=torch.long, device=device)
        logits, _ = model(x)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


@torch.no_grad()
def embedding_similarity(
    model: CharCNN,
    base_texts: list[str],
    aug_texts: list[str],
    vocab: dict[str, int],
    max_len: int,
    device: str,
    batch_size: int,
) -> float:
    model.eval()
    sims: list[float] = []
    for start in range(0, len(base_texts), batch_size):
        base = base_texts[start : start + batch_size]
        aug = aug_texts[start : start + batch_size]
        x1 = torch.tensor([encode(x, vocab, max_len) for x in base], dtype=torch.long, device=device)
        x2 = torch.tensor([encode(x, vocab, max_len) for x in aug], dtype=torch.long, device=device)
        _, h1 = model(x1)
        _, h2 = model(x2)
        sims.extend(torch.cosine_similarity(h1, h2, dim=1).cpu().numpy().tolist())
    return float(np.mean(sims))


def metrics_from_probs(probs: np.ndarray, y: np.ndarray) -> dict[str, float | list[list[int]]]:
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mean_sqli_prob": float(probs[y == 1].mean()) if np.any(y == 1) else math.nan,
        "p10_sqli_prob": float(np.quantile(probs[y == 1], 0.10)) if np.any(y == 1) else math.nan,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def make_test_view(
    test_texts: list[str],
    test_labels: list[int],
    seed: int,
    hard: bool,
    rounds: int,
    mutation_source: str,
    wafamole_repo: str,
) -> tuple[list[str], list[int], list[str], list[str]]:
    texts = []
    base_sqli = []
    aug_sqli = []
    for i, (text, y) in enumerate(zip(test_texts, test_labels)):
        if y == 1:
            aug = mutate_payload(
                text,
                seed=seed * 100_003 + i,
                hard=hard,
                rounds=rounds,
                mutation_source=mutation_source,
                wafamole_repo=wafamole_repo,
            )
            texts.append(aug)
            base_sqli.append(text)
            aug_sqli.append(aug)
        else:
            texts.append(text)
    return texts, list(test_labels), base_sqli, aug_sqli


@torch.no_grad()
def search_evasion(
    model: CharCNN,
    seeds: list[str],
    vocab: dict[str, int],
    max_len: int,
    device: str,
    rng_seed: int,
    rounds: int,
    candidates: int,
    threshold: float,
    mutation_source: str,
    wafamole_repo: str,
    search_mode: str,
    beam_width: int,
    ensure_changed: bool,
    mutation_retries: int,
) -> dict[str, float]:
    rng = random.Random(rng_seed)
    model.eval()
    eligible = 0
    success = 0
    init_probs: list[float] = []
    final_probs: list[float] = []
    query_counts: list[int] = []

    def score(batch: list[str]) -> np.ndarray:
        x = torch.tensor([encode(t, vocab, max_len) for t in batch], dtype=torch.long, device=device)
        logits, _ = model(x)
        return torch.sigmoid(logits).cpu().numpy()

    for seed_text in seeds:
        best_text = seed_text
        best_prob = float(score([best_text])[0])
        queries = 1
        if best_prob < threshold:
            continue
        eligible += 1
        init_probs.append(best_prob)
        beam: list[tuple[str, float]] = [(best_text, best_prob)]
        visited = {best_text}
        for r in range(rounds):
            parents = beam if search_mode == "beam" else beam[:1]
            candidate_texts: list[str] = []
            for parent_rank, (parent_text, _) in enumerate(parents):
                for c in range(candidates):
                    cand = mutate_payload_with_retries(
                        parent_text,
                        seed=rng.randint(0, 2**31 - 1) + r * 10_007 + parent_rank * 997 + c,
                        hard=True,
                        rounds=1 + ((r + c + parent_rank) % 3),
                        mutation_source=mutation_source,
                        wafamole_repo=wafamole_repo,
                        ensure_changed=ensure_changed,
                        retries=mutation_retries,
                    )
                    if cand in visited:
                        continue
                    visited.add(cand)
                    candidate_texts.append(cand)
            if not candidate_texts:
                break
            probs = score(candidate_texts)
            queries += len(candidate_texts)
            ranked = sorted(zip(candidate_texts, probs.tolist()), key=lambda item: item[1])
            if float(ranked[0][1]) < best_prob:
                best_text = ranked[0][0]
                best_prob = float(ranked[0][1])
            if search_mode == "beam":
                merged = {(text, float(prob)) for text, prob in beam}
                merged.update((text, float(prob)) for text, prob in ranked)
                beam = sorted(merged, key=lambda item: item[1])[: max(1, beam_width)]
            else:
                beam = [(best_text, best_prob)]
            if best_prob < threshold:
                break
        final_probs.append(best_prob)
        query_counts.append(float(queries))
        if best_prob < threshold:
            success += 1

    return {
        "eligible": float(eligible),
        "success_rate": float(success / eligible) if eligible else math.nan,
        "init_prob_mean": float(np.mean(init_probs)) if init_probs else math.nan,
        "final_prob_mean": float(np.mean(final_probs)) if final_probs else math.nan,
        "final_prob_median": float(np.median(final_probs)) if final_probs else math.nan,
        "avg_queries": float(np.mean(query_counts)) if query_counts else math.nan,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_summary(rows: list[dict], metric_path: tuple[str, ...], a: str, b: str) -> dict[str, float]:
    by_seed = {}
    for row in rows:
        value = row
        for key in metric_path:
            value = value[key]
        by_seed.setdefault(row["seed"], {})[row["method"]] = float(value)

    diffs = []
    for seed, methods in by_seed.items():
        if a in methods and b in methods:
            diffs.append(methods[b] - methods[a])

    result = {
        "n": float(len(diffs)),
        "mean_diff": float(np.mean(diffs)) if diffs else math.nan,
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
    }
    if len(diffs) >= 3 and np.std(diffs, ddof=1) > 0:
        test = stats.ttest_1samp(diffs, popmean=0.0)
        result["paired_t_p"] = float(test.pvalue)
    else:
        result["paired_t_p"] = math.nan
    return result


def build_train_aug_views(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    args: argparse.Namespace,
) -> list[list[str]]:
    views: list[list[str]] = []
    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        sample_views = []
        if y == 1:
            for v in range(args.num_aug_views):
                sample_views.append(
                    mutate_payload_with_retries(
                        text,
                        seed=seed * 1_000_003 + i * 10_007 + v * 1_009,
                        hard=args.train_aug_hard,
                        rounds=args.train_aug_rounds,
                        mutation_source=args.train_mutation_source or args.mutation_source,
                        wafamole_repo=args.wafamole_repo,
                        ensure_changed=args.ensure_train_aug_changed,
                        retries=args.train_aug_retries,
                    )
                )
        else:
            sample_views = [text] * args.num_aug_views
        views.append(sample_views)
    return views


def summarize_train_aug_views(
    train_texts: list[str],
    train_labels: list[int],
    train_aug_views: list[list[str]],
) -> tuple[int, int, float]:
    total_sqli = 0
    samples_with_change = 0
    changed_view_count = 0
    for text, y, views in zip(train_texts, train_labels, train_aug_views):
        if y != 1:
            continue
        total_sqli += 1
        changed_this = sum(1 for v in views if v != text)
        changed_view_count += changed_this
        if changed_this > 0:
            samples_with_change += 1
    changed_ratio = changed_view_count / total_sqli if total_sqli else 0.0
    return samples_with_change, total_sqli, changed_ratio


def mine_hard_train_aug_views(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
    args: argparse.Namespace,
    warmup_model: CharCNN,
    warmup_vocab: dict[str, int],
    device: str,
) -> list[list[str]]:
    views: list[list[str]] = [[text] * args.num_aug_views for text in train_texts]
    candidate_texts: list[str] = []
    candidate_owner: list[int] = []

    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        if y != 1:
            continue
        for p in range(args.hardmine_pool_size):
            candidate_texts.append(
                mutate_payload_with_retries(
                    text,
                    seed=seed * 10_000_019 + i * 1009 + p * 131,
                    hard=args.hardmine_hard,
                    rounds=args.hardmine_rounds,
                    mutation_source=args.mutation_source,
                    wafamole_repo=args.wafamole_repo,
                    ensure_changed=True,
                    retries=args.hardmine_retries,
                )
            )
            candidate_owner.append(i)

    scores = score_texts(
        warmup_model,
        candidate_texts,
        warmup_vocab,
        args.max_len,
        device,
        args.batch_size,
    )

    ranked_by_owner: dict[int, list[tuple[float, str]]] = {}
    for idx, cand, prob in zip(candidate_owner, candidate_texts, scores.tolist()):
        ranked_by_owner.setdefault(idx, []).append((float(prob), cand))

    for i, (text, y) in enumerate(zip(train_texts, train_labels)):
        if y != 1:
            continue
        ranked = sorted(ranked_by_owner.get(i, []), key=lambda item: item[0])
        chosen: list[str] = []
        seen = {text}
        for _, cand in ranked:
            if cand in seen:
                continue
            seen.add(cand)
            chosen.append(cand)
            if len(chosen) >= args.num_aug_views:
                break
        filler = 0
        while len(chosen) < args.num_aug_views:
            cand = mutate_payload_with_retries(
                text,
                seed=seed * 20_000_033 + i * 2017 + filler * 173,
                hard=args.hardmine_hard,
                rounds=args.hardmine_rounds,
                mutation_source=args.mutation_source,
                wafamole_repo=args.wafamole_repo,
                ensure_changed=True,
                retries=args.hardmine_retries,
            )
            filler += 1
            if cand in seen:
                continue
            seen.add(cand)
            chosen.append(cand)
        views[i] = chosen

    return views


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        try:
            _ = torch.zeros(1, device="mps")
        except RuntimeError as exc:
            print(f"MPS unavailable at runtime ({exc}); falling back to CPU.")
            device = "cpu"

    data_path = ensure_dataset(Path(args.data))
    texts, labels = load_payload_data(data_path, max_len=args.max_len)
    print(f"Loaded payload-level records: total={len(texts)}, benign={labels.count(0)}, sqli={labels.count(1)}")

    methods = args.methods
    rows = []
    started = time.time()

    for seed in args.seeds:
        train_texts, train_labels, test_texts, test_labels = make_split(
            texts,
            labels,
            seed=seed,
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
        )
        train_aug = build_train_aug_views(train_texts, train_labels, seed, args)
        if args.train_view_mining == "hard":
            warmup_vocab = build_vocab(train_texts + [view for views in train_aug for view in views])
            warmup_cfg = TrainConfig(
                method="aug_ce",
                seed=seed,
                epochs=args.hardmine_warmup_epochs,
                batch_size=args.batch_size,
                max_len=args.max_len,
                lr=args.lr,
                consistency_weight=args.consistency_weight,
                device=device,
                canonical_logit_weight=args.canonical_logit_weight,
            )
            print(
                "Warmup hard-mining model seed={} epochs={} pool={} rounds={} hard={}".format(
                    seed,
                    args.hardmine_warmup_epochs,
                    args.hardmine_pool_size,
                    args.hardmine_rounds,
                    args.hardmine_hard,
                )
            )
            warmup_model = train_model(warmup_cfg, train_texts, train_labels, train_aug, warmup_vocab)
            train_aug = mine_hard_train_aug_views(
                train_texts,
                train_labels,
                seed,
                args,
                warmup_model,
                warmup_vocab,
                device,
            )
            del warmup_model
        changed_samples, total_sqli_train, changed_view_ratio = summarize_train_aug_views(
            train_texts, train_labels, train_aug
        )
        print(
            "Train augmentation changed {}/{} SQLi samples; avg changed views per SQLi={:.2f}; rounds={}, hard={}, ensure_changed={}, num_views={}".format(
                changed_samples,
                total_sqli_train,
                changed_view_ratio,
                args.train_aug_rounds,
                args.train_aug_hard,
                args.ensure_train_aug_changed,
                args.num_aug_views,
            )
        )
        test_mut, test_mut_labels, test_base_sqli, test_aug_sqli = make_test_view(
            test_texts,
            test_labels,
            seed=seed + 10_000,
            hard=False,
            rounds=2,
            mutation_source=args.test_mutation_source or args.mutation_source,
            wafamole_repo=args.wafamole_repo,
        )
        test_hard, test_hard_labels, hard_base_sqli, hard_aug_sqli = make_test_view(
            test_texts,
            test_labels,
            seed=seed + 20_000,
            hard=True,
            rounds=3,
            mutation_source=args.test_mutation_source or args.mutation_source,
            wafamole_repo=args.wafamole_repo,
        )
        vocab = build_vocab(train_texts + [view for views in train_aug for view in views])
        evasion_seeds = [text for text, y in zip(test_texts, test_labels) if y == 1][: args.evasion_seeds]

        for method in methods:
            cfg = TrainConfig(
                method=method,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_len=args.max_len,
                lr=args.lr,
                consistency_weight=args.consistency_weight,
                device=device,
            )
            print(f"Training method={method} seed={seed}")
            model = train_model(cfg, train_texts, train_labels, train_aug, vocab)

            clean_probs, clean_y = predict_proba(
                model, test_texts, test_labels, vocab, args.max_len, device, args.batch_size
            )
            mut_probs, mut_y = predict_proba(
                model, test_mut, test_mut_labels, vocab, args.max_len, device, args.batch_size
            )
            hard_probs, hard_y = predict_proba(
                model, test_hard, test_hard_labels, vocab, args.max_len, device, args.batch_size
            )
            sim_mut = embedding_similarity(
                model, test_base_sqli, test_aug_sqli, vocab, args.max_len, device, args.batch_size
            )
            sim_hard = embedding_similarity(
                model, hard_base_sqli, hard_aug_sqli, vocab, args.max_len, device, args.batch_size
            )
            evasion = search_evasion(
                model,
                evasion_seeds,
                vocab,
                args.max_len,
                device,
                rng_seed=seed * 17 + methods.index(method),
                rounds=args.evasion_rounds,
                candidates=args.evasion_candidates,
                threshold=0.5,
                mutation_source=args.test_mutation_source or args.mutation_source,
                wafamole_repo=args.wafamole_repo,
                search_mode=args.evasion_search,
                beam_width=args.evasion_beam_width,
                ensure_changed=args.ensure_evasion_changed,
                mutation_retries=args.evasion_mutation_retries,
            )

            row = {
                "seed": seed,
                "method": method,
                "clean": metrics_from_probs(clean_probs, clean_y),
                "mutated": metrics_from_probs(mut_probs, mut_y),
                "hard_mutated": metrics_from_probs(hard_probs, hard_y),
                "embedding_cosine_mutated": sim_mut,
                "embedding_cosine_hard_mutated": sim_hard,
                "evasion": evasion,
            }
            rows.append(row)
            print(
                "  clean_f1={:.4f} mut_recall={:.4f} hard_recall={:.4f} "
                "hard_p10={:.4f} evasion={:.3f} queries={:.1f} sim_hard={:.4f}".format(
                    row["clean"]["f1"],
                    row["mutated"]["recall"],
                    row["hard_mutated"]["recall"],
                    row["hard_mutated"]["p10_sqli_prob"],
                    row["evasion"]["success_rate"],
                    row["evasion"]["avg_queries"],
                    row["embedding_cosine_hard_mutated"],
                )
            )

    summary = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "clean_f1": summarize([row["clean"]["f1"] for row in method_rows]),
            "mutated_recall": summarize([row["mutated"]["recall"] for row in method_rows]),
            "hard_mutated_recall": summarize([row["hard_mutated"]["recall"] for row in method_rows]),
            "hard_mutated_p10_sqli_prob": summarize([row["hard_mutated"]["p10_sqli_prob"] for row in method_rows]),
            "embedding_cosine_hard_mutated": summarize(
                [row["embedding_cosine_hard_mutated"] for row in method_rows]
            ),
            "evasion_success_rate": summarize([row["evasion"]["success_rate"] for row in method_rows]),
            "evasion_final_prob_mean": summarize([row["evasion"]["final_prob_mean"] for row in method_rows]),
            "evasion_avg_queries": summarize([row["evasion"]["avg_queries"] for row in method_rows]),
        }

    comparisons = {
        "aug_consistency_minus_aug_ce": {
            "hard_mutated_recall": paired_summary(rows, ("hard_mutated", "recall"), "aug_ce", "aug_consistency"),
            "hard_mutated_p10_sqli_prob": paired_summary(
                rows, ("hard_mutated", "p10_sqli_prob"), "aug_ce", "aug_consistency"
            ),
            "embedding_cosine_hard_mutated": paired_summary(
                rows, ("embedding_cosine_hard_mutated",), "aug_ce", "aug_consistency"
            ),
            "evasion_success_rate": paired_summary(
                rows, ("evasion", "success_rate"), "aug_ce", "aug_consistency"
            ),
        }
    }

    result = {
        "config": vars(args) | {"device_resolved": device, "data_url": DATA_URL},
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "summary": summary,
        "comparisons": comparisons,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/SQLiV3_clean.json")
    parser.add_argument("--output", default="experiments/consistency_sqli_results.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--methods", nargs="+", default=["clean_only", "aug_ce", "aug_consistency"])
    parser.add_argument("--mutation-source", choices=["local", "wafamole", "advsqli"], default="local")
    parser.add_argument("--train-mutation-source", choices=["local", "wafamole", "advsqli"], default=None)
    parser.add_argument("--test-mutation-source", choices=["local", "wafamole", "advsqli"], default=None)
    parser.add_argument("--wafamole-repo", default="external/WAF-A-MoLE")
    parser.add_argument("--num-aug-views", type=int, default=1)
    parser.add_argument("--train-view-mining", choices=["random", "hard"], default="random")
    parser.add_argument("--train-aug-rounds", type=int, default=1)
    parser.add_argument("--train-aug-hard", action="store_true")
    parser.add_argument("--ensure-train-aug-changed", action="store_true")
    parser.add_argument("--train-aug-retries", type=int, default=8)
    parser.add_argument("--hardmine-pool-size", type=int, default=12)
    parser.add_argument("--hardmine-rounds", type=int, default=3)
    parser.add_argument("--hardmine-warmup-epochs", type=int, default=3)
    parser.add_argument("--hardmine-hard", action="store_true")
    parser.add_argument("--hardmine-retries", type=int, default=2)
    parser.add_argument("--train-per-class", type=int, default=2500)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--canonical-logit-weight", type=float, default=0.25)
    parser.add_argument("--evasion-seeds", type=int, default=80)
    parser.add_argument("--evasion-rounds", type=int, default=18)
    parser.add_argument("--evasion-candidates", type=int, default=8)
    parser.add_argument("--evasion-search", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--evasion-beam-width", type=int, default=4)
    parser.add_argument("--ensure-evasion-changed", action="store_true")
    parser.add_argument("--evasion-mutation-retries", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
