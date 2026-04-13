#!/usr/bin/env python3
"""Formal dataset loading and splitting helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class DatasetBundle:
    name: str
    path: str
    texts: list[str]
    labels: list[int]
    sources: list[str]
    origins: list[str]


def load_manifest(processed_dir: Path) -> dict:
    return json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))


def load_processed_dataset(path: Path, name: str) -> DatasetBundle:
    rows = json.loads(path.read_text(encoding="utf-8"))
    texts = [str(row["text"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    sources = [str(row.get("source", name)) for row in rows]
    origins = [str(row.get("origin", row.get("source", name))) for row in rows]
    return DatasetBundle(name=name, path=str(path), texts=texts, labels=labels, sources=sources, origins=origins)


def load_dataset_by_name(processed_dir: Path, name: str) -> DatasetBundle:
    manifest = load_manifest(processed_dir)
    rel_path = manifest["datasets"][name]["path"]
    path = (ROOT / rel_path).resolve()
    return load_processed_dataset(path, name)


def class_indices(labels: list[int]) -> dict[int, list[int]]:
    out = {0: [], 1: []}
    for i, y in enumerate(labels):
        if y in out:
            out[y].append(i)
    return out


def make_stratified_split(
    bundle: DatasetBundle,
    seed: int,
    train_per_class: int,
    test_per_class: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    idx = class_indices(bundle.labels)
    rng = random.Random(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for y in [0, 1]:
        pool = idx[y][:]
        rng.shuffle(pool)
        need = train_per_class + test_per_class
        if len(pool) < need:
            raise ValueError(f"{bundle.name} class={y} needs {need}, got {len(pool)}")
        train_idx.extend(pool[:train_per_class])
        test_idx.extend(pool[train_per_class : train_per_class + test_per_class])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    train_texts = [bundle.texts[i] for i in train_idx]
    train_labels = [bundle.labels[i] for i in train_idx]
    test_texts = [bundle.texts[i] for i in test_idx]
    test_labels = [bundle.labels[i] for i in test_idx]
    return train_texts, train_labels, test_texts, test_labels


def filter_dataset(
    bundle: DatasetBundle,
    forbidden_texts: set[str],
) -> DatasetBundle:
    rows = [
        (text, label, source, origin)
        for text, label, source, origin in zip(bundle.texts, bundle.labels, bundle.sources, bundle.origins)
        if text not in forbidden_texts
    ]
    return DatasetBundle(
        name=bundle.name,
        path=bundle.path,
        texts=[row[0] for row in rows],
        labels=[row[1] for row in rows],
        sources=[row[2] for row in rows],
        origins=[row[3] for row in rows],
    )


def balanced_sample(
    bundle: DatasetBundle,
    seed: int,
    per_class: int,
) -> DatasetBundle:
    idx = class_indices(bundle.labels)
    rng = random.Random(seed)
    idx0 = idx[0][:]
    idx1 = idx[1][:]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n = min(len(idx0), len(idx1)) if per_class <= 0 else min(per_class, len(idx0), len(idx1))
    if n <= 0:
        raise ValueError(f"Cannot sample balanced set from {bundle.name}: benign={len(idx0)} sqli={len(idx1)}")
    chosen = idx0[:n] + idx1[:n]
    rng.shuffle(chosen)
    return DatasetBundle(
        name=bundle.name,
        path=bundle.path,
        texts=[bundle.texts[i] for i in chosen],
        labels=[bundle.labels[i] for i in chosen],
        sources=[bundle.sources[i] for i in chosen],
        origins=[bundle.origins[i] for i in chosen],
    )
