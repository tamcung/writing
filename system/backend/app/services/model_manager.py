from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.services.serialization import (
    load_checkpoint,
    save_clean_codebert_checkpoint,
    save_clean_sequence_checkpoint,
    save_pair_codebert_checkpoint,
    save_pair_sequence_checkpoint,
)


settings = get_settings()
if str(settings.root_dir) not in sys.path:
    sys.path.insert(0, str(settings.root_dir))

from experiments.clean_models import CodeBERTConfig, CodeBERTModel, SeqConfig, TextCNNModel, BiLSTMModel  # noqa: E402
from experiments.metrics import metrics_from_probs  # noqa: E402
from experiments.model_utils import load_seed_split, resolve_device, rows_to_xy  # noqa: E402
from experiments.pair_data import load_pair_rows, pair_rows_to_training_arrays  # noqa: E402
from experiments.paired_models import PairCodeBERTConfig, PairCodeBERTModel, PairSeqConfig, PairSequenceModel  # noqa: E402


@dataclass(slots=True)
class ModelSpec:
    key: str
    backbone: str
    method: str
    checkpoint_path: Path
    artifact_type: str = "unknown"
    version: str = "unknown"
    metrics: dict[str, float] = None  # type: ignore[assignment]
    note: str | None = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = {}


class ModelManager:
    def __init__(self, cfg: Settings) -> None:
        self.settings = cfg
        self.device = resolve_device(cfg.model_device)
        self.registry: dict[str, ModelSpec] = {}
        self.loaded_models: dict[str, Any] = {}
        self._bootstrap_if_needed()
        self._discover_checkpoints()
        self._write_manifest()

    def _checkpoint_name(self, backbone: str, method: str) -> Path:
        return self.settings.checkpoint_dir / f"{backbone}_{method}.pt"

    def _enabled_bootstrap_models(self) -> list[str]:
        enabled: list[str] = []
        for model_key in self.settings.bootstrap_models:
            backbone, _ = model_key.split(":", 1)
            if backbone == "codebert" and not self.settings.bootstrap_codebert:
                continue
            enabled.append(model_key)
        return enabled

    def _bootstrap_if_needed(self) -> None:
        if not self.settings.auto_bootstrap:
            return
        required = [self._checkpoint_name(*model_key.split(":", 1)) for model_key in self._enabled_bootstrap_models()]
        missing_required = [path for path in required if not path.exists()]
        if not missing_required and not self.settings.bootstrap_force:
            return
        self.bootstrap_models(force=self.settings.bootstrap_force)

    def bootstrap_models(self, force: bool = False, include_codebert: bool | None = None) -> list[str]:
        include_codebert = self.settings.bootstrap_codebert if include_codebert is None else include_codebert
        outputs: list[str] = []
        train_rows = load_seed_split(self.settings.root_dir / "data" / "splits", self.settings.bootstrap_seed, "train")
        valid_rows = load_seed_split(self.settings.root_dir / "data" / "splits", self.settings.bootstrap_seed, "valid")
        clean_texts, clean_labels = rows_to_xy(train_rows)
        valid_texts, valid_labels = rows_to_xy(valid_rows)

        pair_rows = load_pair_rows(self.settings.root_dir / "data" / "pairs" / f"seed_{self.settings.bootstrap_seed}" / "train_pairs.json")
        pair_canon, pair_mutated, pair_labels = pair_rows_to_training_arrays(pair_rows)

        for model_key in self.settings.bootstrap_models:
            backbone, method = model_key.split(":", 1)
            if backbone == "codebert":
                continue
            checkpoint_path = self._checkpoint_name(backbone, method)
            if checkpoint_path.exists() and not force:
                outputs.append(checkpoint_path.name)
                continue
            metrics: dict[str, float]
            if method == "clean_ce" and backbone in {"textcnn", "bilstm"}:
                cfg = SeqConfig(
                    epochs=self.settings.seq_epochs,
                    batch_size=self.settings.seq_batch_size,
                    max_tokens=self.settings.seq_max_tokens,
                    max_vocab=self.settings.seq_max_vocab,
                    lr=self.settings.seq_lr,
                    emb_dim=self.settings.seq_emb_dim,
                    channels=self.settings.seq_channels,
                    hidden_dim=self.settings.seq_hidden_dim,
                    dropout=self.settings.seq_dropout,
                    lowercase=True,
                    device=self.device,
                )
                model = TextCNNModel(cfg) if backbone == "textcnn" else BiLSTMModel(cfg)
                model.fit(clean_texts, clean_labels)
                probs = model.predict_proba(valid_texts)
                metrics = metrics_from_probs(probs, valid_labels)
                save_clean_sequence_checkpoint(checkpoint_path, backbone, method, model, metrics)
                outputs.append(checkpoint_path.name)
                continue

            if method in {"pair_ce", "pair_canonical"} and backbone in {"textcnn", "bilstm"}:
                cfg = PairSeqConfig(
                    backbone=backbone,
                    method=method,
                    seed=self.settings.bootstrap_seed,
                    epochs=self.settings.seq_epochs,
                    batch_size=self.settings.seq_batch_size,
                    max_tokens=self.settings.seq_max_tokens,
                    max_vocab=self.settings.seq_max_vocab,
                    lr=self.settings.seq_lr,
                    emb_dim=self.settings.seq_emb_dim,
                    channels=self.settings.seq_channels,
                    hidden_dim=self.settings.seq_hidden_dim,
                    dropout=self.settings.seq_dropout,
                    lowercase=True,
                    consistency_weight=self.settings.pair_consistency_weight,
                    hard_align_gamma=self.settings.pair_hard_align_gamma if method == "pair_canonical" else 0.0,
                    device=self.device,
                )
                model = PairSequenceModel(cfg)
                model.fit_pairs(pair_canon, pair_mutated, pair_labels)
                probs = model.predict_proba(valid_texts)
                metrics = metrics_from_probs(probs, valid_labels)
                save_pair_sequence_checkpoint(checkpoint_path, backbone, method, model, metrics)
                outputs.append(checkpoint_path.name)
                continue

        if include_codebert:
            codebert_targets = [item for item in self.settings.bootstrap_models if item.startswith("codebert:")]
            for model_key in codebert_targets:
                backbone, method = model_key.split(":", 1)
                checkpoint_path = self._checkpoint_name(backbone, method)
                if checkpoint_path.exists() and not force:
                    outputs.append(checkpoint_path.name)
                    continue
                if method == "clean_ce":
                    cfg = CodeBERTConfig(
                        model_name=self.settings.codebert_model_name,
                        local_files_only=self.settings.codebert_local_files_only,
                        freeze_encoder=self.settings.codebert_freeze_encoder,
                        epochs=self.settings.codebert_epochs,
                        batch_size=self.settings.codebert_batch_size,
                        max_len=self.settings.codebert_max_len,
                        lr=self.settings.codebert_lr,
                        encoder_lr=self.settings.codebert_encoder_lr,
                        dropout=self.settings.codebert_dropout,
                        device=self.device,
                    )
                    model = CodeBERTModel(cfg)
                    model.fit(clean_texts, clean_labels)
                    probs = model.predict_proba(valid_texts)
                    metrics = metrics_from_probs(probs, valid_labels)
                    save_clean_codebert_checkpoint(checkpoint_path, backbone, method, model, metrics)
                    outputs.append(checkpoint_path.name)
                elif method in {"pair_ce", "pair_canonical"}:
                    cfg = PairCodeBERTConfig(
                        method=method,
                        seed=self.settings.bootstrap_seed,
                        model_name=self.settings.codebert_model_name,
                        local_files_only=self.settings.codebert_local_files_only,
                        freeze_encoder=self.settings.codebert_freeze_encoder,
                        epochs=self.settings.codebert_epochs,
                        batch_size=self.settings.codebert_batch_size,
                        max_len=self.settings.codebert_max_len,
                        lr=self.settings.codebert_lr,
                        encoder_lr=self.settings.codebert_encoder_lr,
                        dropout=self.settings.codebert_dropout,
                        consistency_weight=self.settings.pair_consistency_weight,
                        hard_align_gamma=self.settings.pair_hard_align_gamma if method == "pair_canonical" else 0.0,
                        device=self.device,
                    )
                    model = PairCodeBERTModel(cfg)
                    model.fit_pairs(pair_canon, pair_mutated, pair_labels)
                    probs = model.predict_proba(valid_texts)
                    metrics = metrics_from_probs(probs, valid_labels)
                    save_pair_codebert_checkpoint(checkpoint_path, backbone, method, model, metrics)
                    outputs.append(checkpoint_path.name)

        self._discover_checkpoints()
        self._write_manifest()
        return outputs

    def _write_manifest(self) -> None:
        manifest_path = self.settings.checkpoint_dir / "manifest.json"
        payload = {
            key: {
                "backbone": spec.backbone,
                "method": spec.method,
                "checkpoint_path": str(spec.checkpoint_path),
                "artifact_type": spec.artifact_type,
                "version": spec.version,
                "metrics": spec.metrics,
                "note": spec.note,
            }
            for key, spec in sorted(self.registry.items())
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _discover_checkpoints(self) -> None:
        self.registry.clear()
        for path in sorted(self.settings.checkpoint_dir.glob("*.pt")):
            try:
                payload = load_checkpoint(path, self.device)[1]
            except Exception as exc:
                key = path.stem.replace("_", ":", 1)
                self.registry[key] = ModelSpec(
                    key=key,
                    backbone=path.stem.split("_", 1)[0],
                    method=path.stem.split("_", 1)[1] if "_" in path.stem else "unknown",
                    checkpoint_path=path,
                    note=f"load failed: {exc}",
                )
                continue
            key = f"{payload['backbone']}:{payload['method']}"
            self.registry[key] = ModelSpec(
                key=key,
                backbone=payload["backbone"],
                method=payload["method"],
                checkpoint_path=path,
                artifact_type=payload["artifact_type"],
                version=payload.get("version", "unknown"),
                metrics={k: float(v) for k, v in payload.get("metrics", {}).items() if isinstance(v, (int, float))},
            )

    def list_models(self) -> list[ModelSpec]:
        return [self.registry[key] for key in sorted(self.registry)]

    def get_spec(self, model_key: str) -> ModelSpec:
        if model_key not in self.registry:
            raise KeyError(f"Unknown model key: {model_key}")
        return self.registry[model_key]

    def get_model(self, model_key: str):
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        spec = self.get_spec(model_key)
        model, _ = load_checkpoint(spec.checkpoint_path, self.device)
        self.loaded_models[model_key] = model
        return model

    def predict_proba(self, model_key: str, texts: list[str]) -> list[float]:
        model = self.get_model(model_key)
        probs = model.predict_proba(texts)
        return [float(x) for x in probs]

    def loaded_count(self) -> int:
        return len(self.loaded_models)


@lru_cache(maxsize=1)
def get_model_manager() -> ModelManager:
    return ModelManager(get_settings())
