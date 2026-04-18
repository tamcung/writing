from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.clean_models import BiLSTMModel, BiLSTMNet, CodeBERTClassifier, CodeBERTConfig, CodeBERTModel, SeqConfig, TextCNNModel, TextCNNNet
from experiments.paired_models import PairBiLSTMNet, PairCodeBERTConfig, PairCodeBERTModel, PairSeqConfig, PairSequenceModel, PairTextCNNet, CodeBERTPairNet


def save_clean_sequence_checkpoint(path: Path, backbone: str, method: str, model, metrics: dict[str, float] | None = None) -> None:
    payload = {
        "artifact_type": "clean_sequence",
        "backbone": backbone,
        "method": method,
        "config": asdict(model.cfg),
        "vocab": model.vocab,
        "state_dict": model.model.state_dict(),
        "metrics": metrics or {},
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_pair_sequence_checkpoint(path: Path, backbone: str, method: str, model, metrics: dict[str, float] | None = None) -> None:
    payload = {
        "artifact_type": "pair_sequence",
        "backbone": backbone,
        "method": method,
        "config": asdict(model.cfg),
        "vocab": model.vocab,
        "state_dict": model.model.state_dict(),
        "metrics": metrics or {},
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_clean_codebert_checkpoint(path: Path, backbone: str, method: str, model: CodeBERTModel, metrics: dict[str, float] | None = None) -> None:
    payload = {
        "artifact_type": "clean_codebert",
        "backbone": backbone,
        "method": method,
        "config": asdict(model.cfg),
        "state_dict": model.model.state_dict(),
        "metrics": metrics or {},
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_pair_codebert_checkpoint(path: Path, backbone: str, method: str, model: PairCodeBERTModel, metrics: dict[str, float] | None = None) -> None:
    payload = {
        "artifact_type": "pair_codebert",
        "backbone": backbone,
        "method": method,
        "config": asdict(model.cfg),
        "state_dict": model.model.state_dict() if model.model is not None else {},
        "metrics": metrics or {},
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    artifact_type = payload["artifact_type"]
    backbone = payload["backbone"]
    method = payload["method"]

    if artifact_type == "clean_sequence":
        cfg = SeqConfig(**payload["config"])
        cfg.device = device
        if backbone == "textcnn":
            model = TextCNNModel(cfg)
            net = TextCNNNet(
                vocab_size=max(payload["vocab"].values(), default=1) + 1,
                emb_dim=cfg.emb_dim,
                channels=cfg.channels,
                dropout=cfg.dropout,
            )
        else:
            model = BiLSTMModel(cfg)
            net = BiLSTMNet(
                vocab_size=max(payload["vocab"].values(), default=1) + 1,
                emb_dim=cfg.emb_dim,
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
            )
        net.load_state_dict(payload["state_dict"])
        net.to(device)
        model.model = net
        model.vocab = payload["vocab"]
        return model, payload

    if artifact_type == "pair_sequence":
        cfg = PairSeqConfig(**payload["config"])
        cfg.device = device
        model = PairSequenceModel(cfg)
        vocab_size = max(payload["vocab"].values(), default=1) + 1
        projected = method in {"pair_proj_ce", "pair_canonical"}
        if backbone == "textcnn":
            net = PairTextCNNet(vocab_size, cfg.emb_dim, cfg.channels, cfg.dropout, projected)
        else:
            net = PairBiLSTMNet(vocab_size, cfg.emb_dim, cfg.hidden_dim, cfg.dropout, projected)
        net.load_state_dict(payload["state_dict"])
        net.to(device)
        model.model = net
        model.vocab = payload["vocab"]
        return model, payload

    if artifact_type == "clean_codebert":
        cfg = CodeBERTConfig(**payload["config"])
        cfg.device = device
        model = CodeBERTModel(cfg)
        classifier = CodeBERTClassifier(cfg.model_name, cfg.local_files_only, cfg.dropout)
        classifier.load_state_dict(payload["state_dict"])
        classifier.to(device)
        model.model = classifier
        return model, payload

    if artifact_type == "pair_codebert":
        cfg = PairCodeBERTConfig(**payload["config"])
        cfg.device = device
        model = PairCodeBERTModel(cfg)
        classifier = CodeBERTPairNet(
            model_name=cfg.model_name,
            local_files_only=cfg.local_files_only,
            projected_classifier=cfg.method in {"pair_proj_ce", "pair_canonical"},
            freeze_encoder=cfg.freeze_encoder,
            dropout=cfg.dropout,
        )
        classifier.load_state_dict(payload["state_dict"])
        classifier.to(device)
        model.model = classifier
        return model, payload

    raise ValueError(f"Unsupported artifact_type: {artifact_type}")
