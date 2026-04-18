from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ApplicationProfile, DetectionRecord


def normalize_host(host: str) -> str:
    return host.strip().lower()


def list_applications(db: Session) -> list[ApplicationProfile]:
    stmt = select(ApplicationProfile).order_by(ApplicationProfile.status.desc(), ApplicationProfile.name.asc())
    return list(db.scalars(stmt))


def get_application(db: Session, application_id: int) -> ApplicationProfile | None:
    return db.get(ApplicationProfile, application_id)


def find_application_by_host(db: Session, host: str | None) -> ApplicationProfile | None:
    if not host:
        return None
    normalized = normalize_host(host)
    stmt = select(ApplicationProfile).where(ApplicationProfile.host == normalized)
    return db.scalars(stmt).first()


def create_application(
    db: Session,
    *,
    name: str,
    host: str,
    description: str | None,
    default_model_key: str,
    threshold: float,
    status: str,
) -> ApplicationProfile:
    application = ApplicationProfile(
        name=name.strip(),
        host=normalize_host(host),
        description=description.strip() if description else None,
        default_model_key=default_model_key,
        threshold=threshold,
        status=status,
    )
    db.add(application)
    db.commit()
    db.refresh(application)
    return application


def update_application(
    db: Session,
    application: ApplicationProfile,
    *,
    name: str | None = None,
    host: str | None = None,
    description: str | None = None,
    default_model_key: str | None = None,
    threshold: float | None = None,
    status: str | None = None,
) -> ApplicationProfile:
    if name is not None:
        application.name = name.strip()
    if host is not None:
        application.host = normalize_host(host)
    if description is not None:
        application.description = description.strip() if description else None
    if default_model_key is not None:
        application.default_model_key = default_model_key
    if threshold is not None:
        application.threshold = threshold
    if status is not None:
        application.status = status
    db.add(application)
    db.commit()
    db.refresh(application)
    return application


def delete_application(db: Session, application: ApplicationProfile) -> None:
    db.delete(application)
    db.commit()


def build_application_stats(
    db: Session,
    applications: list[ApplicationProfile],
) -> dict[int, dict[str, int | datetime | None]]:
    stats: dict[int, dict[str, int | datetime | None]] = {
        application.id: {"recent_requests": 0, "recent_alerts": 0, "last_seen_at": None}
        for application in applications
    }
    if not applications:
        return stats

    app_by_id = {application.id: application for application in applications}
    app_by_host = {application.host: application for application in applications}

    stmt = select(DetectionRecord).order_by(DetectionRecord.created_at.desc()).limit(2000)
    for record in db.scalars(stmt):
        metadata = record.metadata_json or {}
        if metadata.get("ingest_source") == "manual_predict":
            continue
        application: ApplicationProfile | None = None
        application_id = metadata.get("application_id")
        if isinstance(application_id, int):
            application = app_by_id.get(application_id)
        if application is None:
            host = normalize_host(str(metadata.get("original_host") or ""))
            application = app_by_host.get(host)
        if application is None:
            continue
        entry = stats[application.id]
        entry["recent_requests"] = int(entry["recent_requests"] or 0) + 1
        if record.predicted_label:
            entry["recent_alerts"] = int(entry["recent_alerts"] or 0) + 1
        created_at = record.created_at
        last_seen_at = entry["last_seen_at"]
        if isinstance(created_at, datetime) and (last_seen_at is None or created_at > last_seen_at):
            entry["last_seen_at"] = created_at
    return stats
