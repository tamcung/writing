#!/usr/bin/env python3
"""Paired robust-training backbones for the formal SQLi experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from experiments.formal.tokenization import build_vocab, encode_tokens


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


class PairSequenceDataset(Dataset):
    def __init__(
        self,
        canon_texts: list[str],
        mutated_texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_tokens: int,
        lowercase: bool,
    ) -> None:
        self.canon_texts = canon_texts
        self.mutated_texts = mutated_texts
        self.labels = labels
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        canon = torch.tensor(
            encode_tokens(self.canon_texts[idx], self.vocab, self.max_tokens, lowercase=self.lowercase),
            dtype=torch.long,
        )
        mutated = torch.tensor(
            encode_tokens(self.mutated_texts[idx], self.vocab, self.max_tokens, lowercase=self.lowercase),
            dtype=torch.long,
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return canon, mutated, label


class PairTextCNNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        channels: int,
        dropout: float,
        projected_classifier: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, channels, kernel_size=3, padding=1),
                nn.Conv1d(emb_dim, channels, kernel_size=4, padding=2),
                nn.Conv1d(emb_dim, channels, kernel_size=5, padding=2),
            ]
        )
        hidden = channels * len(self.convs)
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.classifier = nn.Linear(hidden, 1)
        self.projected_classifier = nn.Linear(hidden, 1)
        self.use_projected_classifier = projected_classifier

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        feats = [torch.max(F.gelu(conv(emb)), dim=2).values for conv in self.convs]
        return self.dropout(torch.cat(feats, dim=1))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.projector(h)

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = self.project(h)
        if self.use_projected_classifier:
            logits = self.projected_classifier(z).squeeze(1)
            rep = z
        else:
            logits = self.classifier(h).squeeze(1)
            rep = h
        return logits, rep, h, z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, rep, _, _ = self.embed(x)
        return logits, rep


class PairBiLSTMNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        projected_classifier: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        hidden = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.classifier = nn.Linear(hidden, 1)
        self.projected_classifier = nn.Linear(hidden, 1)
        self.use_projected_classifier = projected_classifier

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        lengths = x.ne(0).sum(dim=1).clamp_min(1)
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.dropout(torch.cat([h_n[-2], h_n[-1]], dim=1))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.projector(h)

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = self.project(h)
        if self.use_projected_classifier:
            logits = self.projected_classifier(z).squeeze(1)
            rep = z
        else:
            logits = self.classifier(h).squeeze(1)
            rep = h
        return logits, rep, h, z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, rep, _, _ = self.embed(x)
        return logits, rep


@dataclass
class PairSeqConfig:
    backbone: str
    method: str
    seed: int
    epochs: int = 8
    batch_size: int = 128
    max_tokens: int = 128
    max_vocab: int = 20000
    min_freq: int = 1
    lowercase: bool = True
    lr: float = 2e-3
    emb_dim: int = 128
    channels: int = 128
    hidden_dim: int = 128
    dropout: float = 0.25
    consistency_weight: float = 0.1
    canonical_logit_weight: float = 0.0
    hard_align_gamma: float = 0.0
    align_benign_pairs: bool = False
    device: str = "cpu"


class PairSequenceModel:
    def __init__(self, cfg: PairSeqConfig) -> None:
        if cfg.backbone not in {"textcnn", "bilstm"}:
            raise ValueError(f"Unsupported sequence backbone: {cfg.backbone}")
        self.cfg = cfg
        self.vocab: dict[str, int] | None = None
        self.model: nn.Module | None = None

    def fit_pairs(self, canon_texts: list[str], mutated_texts: list[str], labels: list[int]) -> "PairSequenceModel":
        torch.manual_seed(self.cfg.seed)
        self.vocab = build_vocab(
            canon_texts + mutated_texts,
            max_vocab=self.cfg.max_vocab,
            min_freq=self.cfg.min_freq,
            lowercase=self.cfg.lowercase,
        )
        projected = self.cfg.method in {"pair_proj_ce", "pair_canonical"}
        vocab_size = max(self.vocab.values(), default=1) + 1
        if self.cfg.backbone == "textcnn":
            self.model = PairTextCNNet(vocab_size, self.cfg.emb_dim, self.cfg.channels, self.cfg.dropout, projected)
        else:
            self.model = PairBiLSTMNet(vocab_size, self.cfg.emb_dim, self.cfg.hidden_dim, self.cfg.dropout, projected)

        dataset = PairSequenceDataset(
            canon_texts,
            mutated_texts,
            labels,
            self.vocab,
            self.cfg.max_tokens,
            self.cfg.lowercase,
        )
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, generator=generator)
        self.model = self.model.to(self.cfg.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.cfg.epochs):
            for canon_x, mutated_x, y in loader:
                canon_x = canon_x.to(self.cfg.device)
                mutated_x = mutated_x.to(self.cfg.device)
                y = y.to(self.cfg.device)
                opt.zero_grad(set_to_none=True)
                canon_logits, _, _, canon_z = self.model.embed(canon_x)  # type: ignore[union-attr]
                mutated_logits, _, _, mutated_z = self.model.embed(mutated_x)  # type: ignore[union-attr]
                loss = 0.5 * (bce(canon_logits, y) + bce(mutated_logits, y))
                loss = loss + pair_alignment_loss(
                    method=self.cfg.method,
                    y=y,
                    canon_logits=canon_logits,
                    mutated_logits=mutated_logits,
                    canon_z=canon_z,
                    mutated_z=mutated_z,
                    consistency_weight=self.cfg.consistency_weight,
                    canonical_logit_weight=self.cfg.canonical_logit_weight,
                    hard_align_gamma=self.cfg.hard_align_gamma,
                    align_benign_pairs=self.cfg.align_benign_pairs,
                )
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.model is not None and self.vocab is not None
        dataset = PairSequenceDataset(
            texts,
            texts,
            [0] * len(texts),
            self.vocab,
            self.cfg.max_tokens,
            self.cfg.lowercase,
        )
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        self.model.eval()
        probs: list[float] = []
        with torch.inference_mode():
            for x, _, _ in loader:
                x = x.to(self.cfg.device)
                logits, _ = self.model(x)  # type: ignore[misc]
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.asarray(probs, dtype=float)


def pair_alignment_loss(
    method: str,
    y: torch.Tensor,
    canon_logits: torch.Tensor,
    mutated_logits: torch.Tensor,
    canon_z: torch.Tensor,
    mutated_z: torch.Tensor,
    consistency_weight: float,
    canonical_logit_weight: float,
    hard_align_gamma: float,
    align_benign_pairs: bool,
) -> torch.Tensor:
    if method != "pair_canonical":
        return canon_z.new_tensor(0.0)
    mask = torch.ones_like(y, dtype=torch.bool) if align_benign_pairs else y > 0.5
    if not mask.any():
        return canon_z.new_tensor(0.0)

    pair_weights = torch.ones_like(y)
    if hard_align_gamma > 0:
        with torch.no_grad():
            mutated_probs = torch.sigmoid(mutated_logits)
            difficulty = (1.0 - mutated_probs).clamp_min(1e-4).pow(hard_align_gamma)
        pair_weights = pair_weights * difficulty

    loss = consistency_weight * weighted_mean(
        1.0 - F.cosine_similarity(mutated_z[mask], canon_z[mask].detach(), dim=1),
        pair_weights[mask],
    )
    if canonical_logit_weight > 0:
        loss = loss + canonical_logit_weight * weighted_mean(
            (mutated_logits[mask] - canon_logits[mask].detach()).pow(2),
            pair_weights[mask],
        )
    return loss


class CodeBERTPairDataset(Dataset):
    def __init__(
        self,
        canon_texts: list[str],
        mutated_texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int,
    ) -> None:
        self.canon = tokenizer(
            canon_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.mutated = tokenizer(
            mutated_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.canon["input_ids"][idx],
            self.canon["attention_mask"][idx],
            self.mutated["input_ids"][idx],
            self.mutated["attention_mask"][idx],
            self.labels[idx],
        )


class CodeBERTPairNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        local_files_only: bool,
        projected_classifier: bool,
        freeze_encoder: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.classifier = nn.Linear(hidden, 1)
        self.projected_classifier = nn.Linear(hidden, 1)
        self.use_projected_classifier = projected_classifier
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.dropout(outputs.last_hidden_state[:, 0])

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.projector(h)

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(input_ids, attention_mask)
        z = self.project(h)
        if self.use_projected_classifier:
            logits = self.projected_classifier(z).squeeze(1)
            rep = z
        else:
            logits = self.classifier(h).squeeze(1)
            rep = h
        return logits, rep, h, z

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, rep, _, _ = self.embed(input_ids, attention_mask)
        return logits, rep


@dataclass
class PairCodeBERTConfig:
    method: str
    seed: int
    model_name: str = "microsoft/codebert-base"
    local_files_only: bool = True
    freeze_encoder: bool = False
    epochs: int = 2
    batch_size: int = 8
    max_len: int = 320
    lr: float = 1e-3
    encoder_lr: float = 2e-5
    dropout: float = 0.1
    consistency_weight: float = 0.1
    canonical_logit_weight: float = 0.0
    hard_align_gamma: float = 0.0
    align_benign_pairs: bool = False
    grad_clip: float = 1.0
    device: str = "cpu"


class PairCodeBERTModel:
    def __init__(self, cfg: PairCodeBERTConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only)
        self.model: CodeBERTPairNet | None = None

    def _build_optimizer(self, model: CodeBERTPairNet) -> torch.optim.Optimizer:
        if self.cfg.freeze_encoder:
            return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.cfg.lr, weight_decay=1e-4)
        encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        return torch.optim.AdamW(
            [{"params": encoder_params, "lr": self.cfg.encoder_lr}, {"params": head_params, "lr": self.cfg.lr}],
            weight_decay=1e-4,
        )

    def fit_pairs(self, canon_texts: list[str], mutated_texts: list[str], labels: list[int]) -> "PairCodeBERTModel":
        torch.manual_seed(self.cfg.seed)
        projected = self.cfg.method in {"pair_proj_ce", "pair_canonical"}
        self.model = CodeBERTPairNet(
            model_name=self.cfg.model_name,
            local_files_only=self.cfg.local_files_only,
            projected_classifier=projected,
            freeze_encoder=self.cfg.freeze_encoder,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)
        dataset = CodeBERTPairDataset(canon_texts, mutated_texts, labels, self.tokenizer, self.cfg.max_len)
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, generator=generator)
        opt = self._build_optimizer(self.model)
        bce = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.cfg.epochs):
            total_loss = 0.0
            for canon_ids, canon_mask, mutated_ids, mutated_mask, y in loader:
                canon_ids = canon_ids.to(self.cfg.device, non_blocking=self.cfg.device == "cuda")
                canon_mask = canon_mask.to(self.cfg.device, non_blocking=self.cfg.device == "cuda")
                mutated_ids = mutated_ids.to(self.cfg.device, non_blocking=self.cfg.device == "cuda")
                mutated_mask = mutated_mask.to(self.cfg.device, non_blocking=self.cfg.device == "cuda")
                y = y.to(self.cfg.device, non_blocking=self.cfg.device == "cuda")
                opt.zero_grad(set_to_none=True)
                canon_logits, _, _, canon_z = self.model.embed(canon_ids, canon_mask)
                mutated_logits, _, _, mutated_z = self.model.embed(mutated_ids, mutated_mask)
                loss = 0.5 * (bce(canon_logits, y) + bce(mutated_logits, y))
                loss = loss + pair_alignment_loss(
                    method=self.cfg.method,
                    y=y,
                    canon_logits=canon_logits,
                    mutated_logits=mutated_logits,
                    canon_z=canon_z,
                    mutated_z=mutated_z,
                    consistency_weight=self.cfg.consistency_weight,
                    canonical_logit_weight=self.cfg.canonical_logit_weight,
                    hard_align_gamma=self.cfg.hard_align_gamma,
                    align_benign_pairs=self.cfg.align_benign_pairs,
                )
                loss.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                opt.step()
                total_loss += float(loss.detach().cpu())
            print(f"    epoch={epoch + 1}/{self.cfg.epochs} loss={total_loss / max(1, len(loader)):.4f}")
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        probs: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.cfg.batch_size):
                batch_texts = texts[start : start + self.cfg.batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=self.cfg.max_len,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.cfg.device, non_blocking=self.cfg.device == "cuda")
                attention_mask = enc["attention_mask"].to(self.cfg.device, non_blocking=self.cfg.device == "cuda")
                logits, _ = self.model(input_ids, attention_mask)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.asarray(probs, dtype=float)
