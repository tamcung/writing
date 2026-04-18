from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_bootstrap_codebert() -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--codebert-base"
    return cache_dir.exists()


@dataclass(slots=True)
class Settings:
    app_name: str = "SQLi Detection System"
    api_prefix: str = "/api/v1"
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4])
    system_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    backend_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    storage_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(Path(__file__).resolve().parents[2] / 'storage' / 'app.db').as_posix()}",
    ))
    elasticsearch_url: str = field(default_factory=lambda: os.getenv("ELASTICSEARCH_URL", ""))
    cors_origins: list[str] = field(default_factory=lambda: _split_csv(os.getenv("CORS_ORIGINS"), ["http://localhost:5173", "http://127.0.0.1:5173"]))
    auto_bootstrap: bool = field(default_factory=lambda: _as_bool(os.getenv("AUTO_BOOTSTRAP"), default=True))
    bootstrap_force: bool = field(default_factory=lambda: _as_bool(os.getenv("BOOTSTRAP_FORCE"), default=False))
    bootstrap_seed: int = field(default_factory=lambda: int(os.getenv("BOOTSTRAP_SEED", "11")))
    bootstrap_models: list[str] = field(default_factory=lambda: _split_csv(
        os.getenv("BOOTSTRAP_MODELS"),
        [
            "textcnn:clean_ce",
            "textcnn:pair_ce",
            "textcnn:pair_canonical",
            "bilstm:clean_ce",
            "bilstm:pair_ce",
            "bilstm:pair_canonical",
            "codebert:clean_ce",
            "codebert:pair_ce",
        ],
    ))
    bootstrap_codebert: bool = field(default_factory=lambda: _as_bool(os.getenv("BOOTSTRAP_CODEBERT"), default=_default_bootstrap_codebert()))
    seq_epochs: int = field(default_factory=lambda: int(os.getenv("SEQ_EPOCHS", "6")))
    seq_batch_size: int = field(default_factory=lambda: int(os.getenv("SEQ_BATCH_SIZE", "128")))
    seq_max_tokens: int = field(default_factory=lambda: int(os.getenv("SEQ_MAX_TOKENS", "256")))
    seq_max_vocab: int = field(default_factory=lambda: int(os.getenv("SEQ_MAX_VOCAB", "20000")))
    seq_lr: float = field(default_factory=lambda: float(os.getenv("SEQ_LR", "0.002")))
    seq_emb_dim: int = field(default_factory=lambda: int(os.getenv("SEQ_EMB_DIM", "128")))
    seq_channels: int = field(default_factory=lambda: int(os.getenv("SEQ_CHANNELS", "128")))
    seq_hidden_dim: int = field(default_factory=lambda: int(os.getenv("SEQ_HIDDEN_DIM", "128")))
    seq_dropout: float = field(default_factory=lambda: float(os.getenv("SEQ_DROPOUT", "0.25")))
    pair_consistency_weight: float = field(default_factory=lambda: float(os.getenv("PAIR_CONSISTENCY_WEIGHT", "0.1")))
    pair_hard_align_gamma: float = field(default_factory=lambda: float(os.getenv("PAIR_HARD_ALIGN_GAMMA", "0.5")))
    codebert_model_name: str = field(default_factory=lambda: os.getenv("CODEBERT_MODEL_NAME", "microsoft/codebert-base"))
    codebert_local_files_only: bool = field(default_factory=lambda: _as_bool(os.getenv("CODEBERT_LOCAL_ONLY"), default=True))
    codebert_freeze_encoder: bool = field(default_factory=lambda: _as_bool(os.getenv("CODEBERT_FREEZE_ENCODER"), default=False))
    codebert_epochs: int = field(default_factory=lambda: int(os.getenv("CODEBERT_EPOCHS", "1")))
    codebert_batch_size: int = field(default_factory=lambda: int(os.getenv("CODEBERT_BATCH_SIZE", "4")))
    codebert_max_len: int = field(default_factory=lambda: int(os.getenv("CODEBERT_MAX_LEN", "320")))
    codebert_lr: float = field(default_factory=lambda: float(os.getenv("CODEBERT_LR", "0.001")))
    codebert_encoder_lr: float = field(default_factory=lambda: float(os.getenv("CODEBERT_ENCODER_LR", "0.00002")))
    codebert_dropout: float = field(default_factory=lambda: float(os.getenv("CODEBERT_DROPOUT", "0.1")))
    model_device: str = field(default_factory=lambda: os.getenv("MODEL_DEVICE", "auto"))
    default_threshold: float = field(default_factory=lambda: float(os.getenv("DEFAULT_THRESHOLD", "0.5")))
    default_decode_passes: int = field(default_factory=lambda: int(os.getenv("DEFAULT_DECODE_PASSES", "1")))

    def __post_init__(self) -> None:
        self.storage_dir = self.backend_dir / "storage"
        self.checkpoint_dir = self.storage_dir / "checkpoints"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
