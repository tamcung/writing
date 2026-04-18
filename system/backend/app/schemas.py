from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    database: str
    elasticsearch: str
    models_loaded: int
    models_available: int


class ModelCard(BaseModel):
    key: str
    backbone: str
    method: str
    artifact_type: str
    status: str
    version: str
    checkpoint_path: str
    metrics: dict[str, float] = Field(default_factory=dict)
    note: str | None = None


class ApplicationProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    host: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1000)
    default_model_key: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    status: str = Field(default="enabled")


class ApplicationProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    host: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1000)
    default_model_key: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    status: str | None = None


class ApplicationProfileOut(BaseModel):
    id: int
    name: str
    host: str
    description: str | None = None
    default_model_key: str
    threshold: float
    status: str
    recent_requests: int = 0
    recent_alerts: int = 0
    last_seen_at: datetime | None = None
    ingress_url: str
    created_at: datetime
    updated_at: datetime


class ExtractedParameter(BaseModel):
    name: str
    source: str
    raw_value: str
    decoded_value: str


class PredictRequest(BaseModel):
    raw_input: str = Field(..., min_length=1, description="Full HTTP request, query string, URL, or raw payload.")
    model_key: str
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    persist: bool = True


class PredictResponse(BaseModel):
    request_id: str
    model_key: str
    backbone: str
    method: str
    probability: float
    predicted_label: int
    threshold: float
    risk_level: str
    normalized_text: str
    extracted_params: list[ExtractedParameter]
    record_id: int | None = None
    model_version: str
    timing_ms: float


class BatchItem(BaseModel):
    item_id: str
    raw_input: str


class BatchPredictRequest(BaseModel):
    model_key: str
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    items: list[BatchItem] = Field(..., min_length=1, max_length=128)


class BatchPredictRow(BaseModel):
    item_id: str
    probability: float
    predicted_label: int
    risk_level: str
    normalized_text: str


class BatchPredictResponse(BaseModel):
    model_key: str
    results: list[BatchPredictRow]


class DetectionRecordOut(BaseModel):
    id: int
    request_id: str
    model_key: str
    backbone: str
    method: str
    model_version: str
    probability: float
    predicted_label: int
    risk_level: str
    normalized_text: str
    raw_input: str
    threshold: float
    timing_ms: float
    matched_params: list[dict[str, Any]]
    metadata_json: dict[str, Any]
    created_at: datetime


class SummaryCount(BaseModel):
    name: str
    count: int


class TrendPoint(BaseModel):
    label: str
    total: int
    malicious: int


class OverviewMetrics(BaseModel):
    total_requests: int
    malicious_requests: int
    high_risk_requests: int
    monitored_hosts: int
    models_available: int


class OverviewResponse(BaseModel):
    metrics: OverviewMetrics
    hourly_trend: list[TrendPoint] = Field(default_factory=list)
    host_distribution: list[SummaryCount] = Field(default_factory=list)
    risk_distribution: list[SummaryCount] = Field(default_factory=list)
    uri_distribution: list[SummaryCount] = Field(default_factory=list)
    recent_alerts: list[DetectionRecordOut] = Field(default_factory=list)


OverviewResponse.model_rebuild()
