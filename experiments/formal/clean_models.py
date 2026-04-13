#!/usr/bin/env python3
"""Clean baseline backbones for the formal experiment suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from experiments.formal.tokenization import build_vocab, encode_tokens, tokenize_sql


class WordSVCModel:
    def __init__(self, ngram_max: int = 2, min_df: int = 1, c: float = 1.0) -> None:
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda text: tokenize_sql(text, lowercase=True),
            token_pattern=None,
            lowercase=False,
            analyzer="word",
            ngram_range=(1, ngram_max),
            min_df=min_df,
            sublinear_tf=True,
        )
        self.classifier = CalibratedClassifierCV(LinearSVC(C=c), method="sigmoid", cv=3)

    def fit(self, texts: list[str], labels: list[int]) -> "WordSVCModel":
        x = self.vectorizer.fit_transform(texts)
        self.classifier.fit(x, labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(x)[:, 1]


class SequenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_tokens: int, lowercase: bool) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(
            encode_tokens(self.texts[idx], self.vocab, self.max_tokens, lowercase=self.lowercase),
            dtype=torch.long,
        )
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class TextCNNNet(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, channels, kernel_size=3, padding=1),
                nn.Conv1d(emb_dim, channels, kernel_size=4, padding=2),
                nn.Conv1d(emb_dim, channels, kernel_size=5, padding=2),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(self.convs), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        feats = [torch.max(F.gelu(conv(emb)), dim=2).values for conv in self.convs]
        h = self.dropout(torch.cat(feats, dim=1))
        return self.fc(h).squeeze(1)


class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = x.ne(0).sum(dim=1).clamp_min(1)
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h).squeeze(1)


@dataclass
class SeqConfig:
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
    device: str = "cpu"


def _train_sequence_model(
    model: nn.Module,
    texts: list[str],
    labels: list[int],
    vocab: dict[str, int],
    cfg: SeqConfig,
) -> nn.Module:
    ds = SequenceDataset(texts, labels, vocab, cfg.max_tokens, cfg.lowercase)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(cfg.epochs):
        for x, y in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, y)
            loss.backward()
            opt.step()
    return model


def _predict_sequence_model(
    model: nn.Module,
    vocab: dict[str, int],
    texts: list[str],
    cfg: SeqConfig,
) -> np.ndarray:
    ds = SequenceDataset(texts, [0] * len(texts), vocab, cfg.max_tokens, cfg.lowercase)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(cfg.device)
            logits = model(x)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


class TextCNNModel:
    def __init__(self, cfg: SeqConfig) -> None:
        self.cfg = cfg
        self.model: TextCNNNet | None = None
        self.vocab: dict[str, int] | None = None

    def fit(self, texts: list[str], labels: list[int]) -> "TextCNNModel":
        self.vocab = build_vocab(
            texts,
            max_vocab=self.cfg.max_vocab,
            min_freq=self.cfg.min_freq,
            lowercase=self.cfg.lowercase,
        )
        net = TextCNNNet(
            vocab_size=max(self.vocab.values(), default=1) + 1,
            emb_dim=self.cfg.emb_dim,
            channels=self.cfg.channels,
            dropout=self.cfg.dropout,
        )
        self.model = _train_sequence_model(net, texts, labels, self.vocab, self.cfg)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.model is not None and self.vocab is not None
        return _predict_sequence_model(self.model, self.vocab, texts, self.cfg)


class BiLSTMModel:
    def __init__(self, cfg: SeqConfig) -> None:
        self.cfg = cfg
        self.model: BiLSTMNet | None = None
        self.vocab: dict[str, int] | None = None

    def fit(self, texts: list[str], labels: list[int]) -> "BiLSTMModel":
        self.vocab = build_vocab(
            texts,
            max_vocab=self.cfg.max_vocab,
            min_freq=self.cfg.min_freq,
            lowercase=self.cfg.lowercase,
        )
        net = BiLSTMNet(
            vocab_size=max(self.vocab.values(), default=1) + 1,
            emb_dim=self.cfg.emb_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        )
        self.model = _train_sequence_model(net, texts, labels, self.vocab, self.cfg)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.model is not None and self.vocab is not None
        return _predict_sequence_model(self.model, self.vocab, texts, self.cfg)


class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name: str, local_files_only: bool, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            h = outputs.pooler_output
        else:
            h = outputs.last_hidden_state[:, 0]
        h = self.dropout(h)
        return self.fc(h).squeeze(1)


@dataclass
class CodeBERTConfig:
    model_name: str = "microsoft/codebert-base"
    local_files_only: bool = True
    freeze_encoder: bool = False
    epochs: int = 2
    batch_size: int = 8
    max_len: int = 320
    lr: float = 1e-3
    encoder_lr: float = 2e-5
    dropout: float = 0.1
    device: str = "cpu"


class CodeBERTDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class CodeBERTModel:
    def __init__(self, cfg: CodeBERTConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only)
        self.model = CodeBERTClassifier(cfg.model_name, cfg.local_files_only, cfg.dropout).to(cfg.device)
        if cfg.freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

    def fit(self, texts: list[str], labels: list[int]) -> "CodeBERTModel":
        ds = CodeBERTDataset(texts, labels, self.tokenizer, self.cfg.max_len)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
        params = []
        if encoder_params:
            params.append({"params": encoder_params, "lr": self.cfg.encoder_lr})
        params.append({"params": head_params, "lr": self.cfg.lr})
        opt = torch.optim.AdamW(params)
        bce = nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(self.cfg.epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.cfg.device)
                attention_mask = batch["attention_mask"].to(self.cfg.device)
                labels = batch["labels"].to(self.cfg.device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(input_ids, attention_mask)
                loss = bce(logits, labels)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        ds = CodeBERTDataset(texts, [0] * len(texts), self.tokenizer, self.cfg.max_len)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False)
        self.model.eval()
        probs: list[float] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.cfg.device)
                attention_mask = batch["attention_mask"].to(self.cfg.device)
                logits = self.model(input_ids, attention_mask)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.asarray(probs, dtype=float)
