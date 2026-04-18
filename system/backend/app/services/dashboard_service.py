from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ApplicationProfile, DetectionRecord


def _to_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_detection_records(db: Session) -> list[DetectionRecord]:
    stmt = select(DetectionRecord).order_by(DetectionRecord.created_at.desc())
    return list(db.scalars(stmt))


def _ingest_source(record: DetectionRecord) -> str:
    metadata = record.metadata_json or {}
    return str(metadata.get("ingest_source") or "manual_predict")


def _is_external_record(record: DetectionRecord) -> bool:
    return _ingest_source(record) != "manual_predict"


def _ingress_node(record: DetectionRecord) -> str:
    metadata = record.metadata_json or {}
    return str(metadata.get("ingress_node") or metadata.get("traffic_gateway") or "")


def _host_label(record: DetectionRecord, application_by_host: dict[str, str]) -> str:
    metadata = record.metadata_json or {}
    application_name = metadata.get("application_name")
    if application_name:
        return str(application_name)
    original_host = str(metadata.get("original_host") or "")
    if original_host and original_host.lower() in application_by_host:
        return application_by_host[original_host.lower()]
    return str(original_host or _ingress_node(record) or "unknown")


def _uri_label(record: DetectionRecord) -> str:
    metadata = record.metadata_json or {}
    original_uri = str(metadata.get("original_uri") or "")
    if not original_uri:
        return "unknown"
    parsed = urlsplit(original_uri)
    return parsed.path or original_uri or "unknown"


def build_overview_payload(
    *,
    records: list[DetectionRecord],
    applications: list[ApplicationProfile],
    models_available: int,
) -> dict:
    traffic_records = [record for record in records if _is_external_record(record)]
    application_by_host = {application.host.lower(): application.name for application in applications}
    enabled_application_count = sum(1 for application in applications if application.status == "enabled")
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=23)
    hourly_index: dict[str, dict[str, int]] = {}
    for offset in range(24):
        current = window_start + timedelta(hours=offset)
        label = current.strftime("%H:00")
        hourly_index[label] = {"label": label, "total": 0, "malicious": 0}

    host_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    uri_counter: Counter[str] = Counter()

    for record in traffic_records:
        host_counter[_host_label(record, application_by_host)] += 1
        risk_counter[record.risk_level] += 1
        if record.predicted_label or record.risk_level in {"high", "critical"}:
            uri_counter[_uri_label(record)] += 1

        created_at = _to_utc(record.created_at)
        if created_at is None or created_at < window_start:
            continue
        label = created_at.strftime("%H:00")
        bucket = hourly_index.get(label)
        if bucket is None:
            continue
        bucket["total"] += 1
        if record.predicted_label:
            bucket["malicious"] += 1

    recent_alerts = [record for record in traffic_records if record.predicted_label][:8]
    if not uri_counter:
        for record in traffic_records:
            uri_counter[_uri_label(record)] += 1

    return {
        "metrics": {
            "total_requests": len(traffic_records),
            "malicious_requests": sum(1 for record in traffic_records if record.predicted_label),
            "high_risk_requests": sum(1 for record in traffic_records if record.risk_level in {"high", "critical"}),
            "monitored_hosts": enabled_application_count,
            "models_available": models_available,
        },
        "hourly_trend": list(hourly_index.values()),
        "host_distribution": [{"name": name, "count": count} for name, count in host_counter.most_common()],
        "risk_distribution": [{"name": name, "count": count} for name, count in risk_counter.most_common()],
        "uri_distribution": [{"name": name, "count": count} for name, count in uri_counter.most_common(8)],
        "recent_alerts": recent_alerts,
    }
