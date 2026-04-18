from __future__ import annotations

import uuid

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from app.core.elasticsearch import get_elasticsearch_client
from app.models import DetectionRecord


def create_detection_record(
    db: Session,
    *,
    model_key: str,
    backbone: str,
    method: str,
    raw_input: str,
    normalized_text: str,
    decode_passes: int,
    threshold: float,
    timing_ms: float,
    probability: float,
    predicted_label: int,
    risk_level: str,
    matched_params: list[dict],
    model_version: str,
    metadata: dict,
) -> DetectionRecord:
    record = DetectionRecord(
        request_id=uuid.uuid4().hex[:16],
        model_key=model_key,
        backbone=backbone,
        method=method,
        raw_input=raw_input,
        normalized_text=normalized_text,
        decode_passes=decode_passes,
        threshold=threshold,
        timing_ms=timing_ms,
        probability=probability,
        predicted_label=predicted_label,
        risk_level=risk_level,
        matched_params=matched_params,
        model_version=model_version,
        metadata_json=metadata,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    client = get_elasticsearch_client()
    if client is not None:
        try:
            client.index(
                index="sqli-detections",
                id=str(record.id),
                document={
                    "record_id": record.id,
                    "request_id": record.request_id,
                    "model_key": record.model_key,
                    "backbone": record.backbone,
                    "method": record.method,
                    "probability": record.probability,
                    "predicted_label": record.predicted_label,
                    "risk_level": record.risk_level,
                    "normalized_text": record.normalized_text,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                },
            )
        except Exception:
            pass
    return record


def list_detection_records(
    db: Session,
    *,
    limit: int = 50,
    model_key: str | None = None,
    ingest_source: str | None = None,
    risk_level: str | None = None,
    predicted_label: int | None = None,
    search: str | None = None,
) -> list[DetectionRecord]:
    stmt: Select[tuple[DetectionRecord]] = select(DetectionRecord)
    if model_key:
        stmt = stmt.where(DetectionRecord.model_key == model_key)
    if risk_level:
        stmt = stmt.where(DetectionRecord.risk_level == risk_level)
    if predicted_label is not None:
        stmt = stmt.where(DetectionRecord.predicted_label == predicted_label)
    if ingest_source:
        if ingest_source == "external":
            stmt = stmt.where(DetectionRecord.metadata_json["ingest_source"].as_string() != "manual_predict")
        else:
            stmt = stmt.where(DetectionRecord.metadata_json["ingest_source"].as_string() == ingest_source)
    if search:
        keyword = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                DetectionRecord.request_id.ilike(keyword),
                DetectionRecord.model_key.ilike(keyword),
                DetectionRecord.normalized_text.ilike(keyword),
                DetectionRecord.raw_input.ilike(keyword),
                DetectionRecord.metadata_json["original_uri"].as_string().ilike(keyword),
            )
        )
    stmt = stmt.order_by(DetectionRecord.created_at.desc()).limit(limit)
    return list(db.scalars(stmt))
