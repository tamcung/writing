from __future__ import annotations

import time
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import engine, get_db
from app.core.elasticsearch import get_elasticsearch_client
from app.models import ApplicationProfile, DetectionRecord
from app.schemas import (
    ApplicationProfileCreate,
    ApplicationProfileOut,
    ApplicationProfileUpdate,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictRow,
    DetectionRecordOut,
    HealthResponse,
    ModelCard,
    OverviewResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.application_service import (
    build_application_stats,
    create_application,
    delete_application,
    find_application_by_host,
    get_application,
    list_applications,
    update_application,
)
from app.services.dashboard_service import build_overview_payload, load_detection_records
from app.services.history_service import create_detection_record, list_detection_records
from app.services.model_manager import ModelManager, get_model_manager
from app.services.preprocessing import ParameterMatch, preprocess_raw_input, risk_level


router = APIRouter()
settings = get_settings()


def _application_to_schema(application: ApplicationProfile, *, stats: dict | None = None) -> ApplicationProfileOut:
    summary = stats or {}
    return ApplicationProfileOut(
        id=application.id,
        name=application.name,
        host=application.host,
        description=application.description,
        default_model_key=application.default_model_key,
        threshold=application.threshold,
        status=application.status,
        recent_requests=int(summary.get("recent_requests") or 0),
        recent_alerts=int(summary.get("recent_alerts") or 0),
        last_seen_at=summary.get("last_seen_at"),
        ingress_url=f"{settings.api_prefix}/ingest/traffic?application_id={application.id}",
        created_at=application.created_at,
        updated_at=application.updated_at,
    )


def _build_response(
    *,
    db: Session,
    manager: ModelManager,
    model_key: str,
    raw_input: str,
    normalized_text: str,
    params: list[ParameterMatch],
    decode_passes: int,
    threshold: float,
    persist: bool,
    metadata: dict,
) -> PredictResponse:
    start = time.perf_counter()
    spec = manager.get_spec(model_key)
    probability = manager.predict_proba(model_key, [normalized_text])[0]
    predicted_label = int(probability >= threshold)
    level = risk_level(probability, threshold)
    timing_ms = (time.perf_counter() - start) * 1000
    record = None
    request_id = ""
    if persist:
        record = create_detection_record(
            db,
            model_key=model_key,
            backbone=spec.backbone,
            method=spec.method,
            raw_input=raw_input,
            normalized_text=normalized_text,
            decode_passes=decode_passes,
            threshold=threshold,
            timing_ms=timing_ms,
            probability=probability,
            predicted_label=predicted_label,
            risk_level=level,
            matched_params=[param.to_dict() for param in params],
            model_version=spec.version,
            metadata={"artifact_type": spec.artifact_type, **metadata},
        )
        request_id = record.request_id
    else:
        import uuid
        request_id = uuid.uuid4().hex[:16]

    return PredictResponse(
        request_id=request_id,
        model_key=model_key,
        backbone=spec.backbone,
        method=spec.method,
        probability=probability,
        predicted_label=predicted_label,
        threshold=threshold,
        risk_level=level,
        normalized_text=normalized_text,
        extracted_params=[param.to_dict() for param in params],
        record_id=record.id if record is not None else None,
        model_version=spec.version,
        timing_ms=timing_ms,
    )


def _record_to_schema(record: DetectionRecord) -> DetectionRecordOut:
    return DetectionRecordOut(
        id=record.id,
        request_id=record.request_id,
        model_key=record.model_key,
        backbone=record.backbone,
        method=record.method,
        model_version=record.model_version,
        probability=record.probability,
        predicted_label=record.predicted_label,
        risk_level=record.risk_level,
        normalized_text=record.normalized_text,
        raw_input=record.raw_input,
        threshold=record.threshold,
        timing_ms=record.timing_ms,
        matched_params=record.matched_params,
        metadata_json=record.metadata_json,
        created_at=record.created_at,
    )


def _health_payload(manager: ModelManager) -> dict[str, str | int]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "up"
    except Exception:
        db_status = "down"
    es_client = get_elasticsearch_client()
    es_status = "disabled"
    if es_client is not None:
        try:
            es_status = "up" if es_client.ping() else "down"
        except Exception:
            es_status = "down"
    return {
        "status": "ok",
        "database": db_status,
        "elasticsearch": es_status,
        "models_loaded": manager.loaded_count(),
        "models_available": len(manager.list_models()),
    }


def _reconstruct_traffic_request(request: Request, raw_body: str) -> tuple[str, dict]:
    headers = request.headers
    original_method = headers.get("x-original-method", request.method)
    original_uri = headers.get("x-original-uri", request.url.path)
    original_host = headers.get("x-original-host", headers.get("host", "upstream.local"))
    original_remote_addr = headers.get("x-original-remote-addr", request.client.host if request.client else "")
    original_scheme = headers.get("x-original-scheme", "http")
    original_user_agent = headers.get("x-original-user-agent", headers.get("user-agent", "traffic-gateway"))
    content_type = headers.get("content-type", "")
    request_line = f"{original_method} {original_uri} HTTP/1.1"
    http_headers = [f"Host: {original_host}", f"User-Agent: {original_user_agent}"]
    if content_type:
        http_headers.append(f"Content-Type: {content_type}")
    if original_remote_addr:
        http_headers.append(f"X-Forwarded-For: {original_remote_addr}")
    raw_input = "\n".join([request_line, *http_headers, "", raw_body])
    gateway_node = headers.get("x-traffic-node", "nginx")
    metadata = {
        "ingest_source": "traffic_gateway",
        "ingress_node": gateway_node,
        "traffic_gateway": gateway_node,
        "collection_mode": headers.get("x-traffic-mode", "request_copy"),
        "original_method": original_method,
        "original_uri": original_uri,
        "original_host": original_host,
        "original_remote_addr": original_remote_addr,
        "original_scheme": original_scheme,
    }
    return raw_input.strip(), metadata


@router.get("/health", response_model=HealthResponse)
def health(manager: ModelManager = Depends(get_model_manager)) -> HealthResponse:
    return HealthResponse(**_health_payload(manager))


@router.get("/models", response_model=list[ModelCard])
def list_models_endpoint(manager: ModelManager = Depends(get_model_manager)) -> list[ModelCard]:
    models = []
    for spec in manager.list_models():
        status = "loaded" if spec.key in manager.loaded_models else ("error" if spec.note else "ready")
        models.append(
            ModelCard(
                key=spec.key,
                backbone=spec.backbone,
                method=spec.method,
                artifact_type=spec.artifact_type,
                status=status,
                version=spec.version,
                checkpoint_path=str(spec.checkpoint_path),
                metrics=spec.metrics,
                note=spec.note,
            )
        )
    return models


@router.get("/applications", response_model=list[ApplicationProfileOut])
def list_applications_endpoint(db: Session = Depends(get_db)) -> list[ApplicationProfileOut]:
    applications = list_applications(db)
    stats = build_application_stats(db, applications)
    return [_application_to_schema(application, stats=stats.get(application.id)) for application in applications]


@router.post("/applications", response_model=ApplicationProfileOut)
def create_application_endpoint(
    payload: ApplicationProfileCreate,
    db: Session = Depends(get_db),
    manager: ModelManager = Depends(get_model_manager),
) -> ApplicationProfileOut:
    try:
        manager.get_spec(payload.default_model_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        application = create_application(
            db,
            name=payload.name,
            host=payload.host,
            description=payload.description,
            default_model_key=payload.default_model_key,
            threshold=payload.threshold,
            status=payload.status,
        )
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Application name or host already exists.") from exc
    return _application_to_schema(application)


@router.patch("/applications/{application_id}", response_model=ApplicationProfileOut)
def update_application_endpoint(
    application_id: int,
    payload: ApplicationProfileUpdate,
    db: Session = Depends(get_db),
    manager: ModelManager = Depends(get_model_manager),
) -> ApplicationProfileOut:
    application = get_application(db, application_id)
    if application is None:
        raise HTTPException(status_code=404, detail="Application not found.")
    if payload.default_model_key:
        try:
            manager.get_spec(payload.default_model_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        application = update_application(
            db,
            application,
            name=payload.name,
            host=payload.host,
            description=payload.description,
            default_model_key=payload.default_model_key,
            threshold=payload.threshold,
            status=payload.status,
        )
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Application name or host already exists.") from exc
    stats = build_application_stats(db, [application])
    return _application_to_schema(application, stats=stats.get(application.id))


@router.delete("/applications/{application_id}")
def delete_application_endpoint(application_id: int, db: Session = Depends(get_db)) -> dict[str, bool]:
    application = get_application(db, application_id)
    if application is None:
        raise HTTPException(status_code=404, detail="Application not found.")
    delete_application(db, application)
    return {"ok": True}


@router.get("/overview", response_model=OverviewResponse)
def overview(
    db: Session = Depends(get_db),
    manager: ModelManager = Depends(get_model_manager),
) -> OverviewResponse:
    records = load_detection_records(db)
    applications = list_applications(db)
    payload = build_overview_payload(records=records, applications=applications, models_available=len(manager.list_models()))
    payload["recent_alerts"] = [_record_to_schema(record) for record in payload["recent_alerts"]]
    return OverviewResponse(**payload)


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    manager: ModelManager = Depends(get_model_manager),
) -> PredictResponse:
    try:
        manager.get_spec(payload.model_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    prepared = preprocess_raw_input(payload.raw_input, settings.default_decode_passes)
    if not prepared.normalized_text:
        raise HTTPException(status_code=400, detail="Failed to extract a valid parameter value or payload text.")
    threshold = payload.threshold if payload.threshold is not None else settings.default_threshold
    return _build_response(
        db=db,
        manager=manager,
        model_key=payload.model_key,
        raw_input=payload.raw_input,
        normalized_text=prepared.normalized_text,
        params=prepared.params,
        decode_passes=settings.default_decode_passes,
        threshold=threshold,
        persist=payload.persist,
        metadata={"ingest_source": "manual_predict"},
    )


@router.post("/batch-predict", response_model=BatchPredictResponse)
def batch_predict(payload: BatchPredictRequest, manager: ModelManager = Depends(get_model_manager)) -> BatchPredictResponse:
    try:
        manager.get_spec(payload.model_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    threshold = payload.threshold if payload.threshold is not None else settings.default_threshold
    normalized_rows: list[tuple[str, str]] = []
    for item in payload.items:
        prepared = preprocess_raw_input(item.raw_input, settings.default_decode_passes)
        normalized_rows.append((item.item_id, prepared.normalized_text))
    probs = manager.predict_proba(payload.model_key, [row[1] for row in normalized_rows])
    results = []
    for (item_id, normalized_text), probability in zip(normalized_rows, probs):
        predicted_label = int(probability >= threshold)
        results.append(
            BatchPredictRow(
                item_id=item_id,
                probability=probability,
                predicted_label=predicted_label,
                risk_level=risk_level(probability, threshold),
                normalized_text=normalized_text,
            )
        )
    return BatchPredictResponse(model_key=payload.model_key, results=results)


@router.get("/traffic-records", response_model=list[DetectionRecordOut])
def traffic_records(
    limit: int = Query(default=50, ge=1, le=500),
    model_key: str | None = Query(default=None),
    ingest_source: str | None = Query(default=None),
    risk_level: str | None = Query(default=None),
    predicted_label: int | None = Query(default=None, ge=0, le=1),
    search: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[DetectionRecordOut]:
    records = list_detection_records(
        db,
        limit=limit,
        model_key=model_key,
        ingest_source=ingest_source,
        risk_level=risk_level,
        predicted_label=predicted_label,
        search=search,
    )
    return [_record_to_schema(record) for record in records]


@router.api_route("/ingest/traffic", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def ingest_traffic(
    request: Request,
    application_id: int | None = Query(default=None),
    model_key: str | None = Query(default=None),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    persist: bool = Query(default=True),
    db: Session = Depends(get_db),
    manager: ModelManager = Depends(get_model_manager),
) -> PredictResponse:
    body = (await request.body()).decode("utf-8", errors="replace")
    raw_input, metadata = _reconstruct_traffic_request(request, body)
    application = None
    if application_id is not None:
        application = get_application(db, application_id)
        if application is None:
            raise HTTPException(status_code=404, detail="Application not found.")
    else:
        application = find_application_by_host(db, str(metadata.get("original_host") or ""))

    if application is not None:
        if application.status != "enabled":
            raise HTTPException(status_code=403, detail="Application is disabled.")
        metadata["application_id"] = application.id
        metadata["application_name"] = application.name
        metadata["application_host"] = application.host

    resolved_model_key = model_key or (application.default_model_key if application is not None else "textcnn:pair_canonical")
    resolved_threshold = threshold if threshold is not None else (application.threshold if application is not None else settings.default_threshold)

    try:
        manager.get_spec(resolved_model_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    prepared = preprocess_raw_input(raw_input or body or request.url.path, settings.default_decode_passes)
    if not prepared.normalized_text:
        fallback_query = request.headers.get("x-original-query", "")
        query_text = fallback_query or urlencode(dict(request.query_params))
        prepared = preprocess_raw_input(query_text, settings.default_decode_passes)
    if not prepared.normalized_text:
        raise HTTPException(status_code=400, detail="Traffic request does not contain detectable parameter values.")

    return _build_response(
        db=db,
        manager=manager,
        model_key=resolved_model_key,
        raw_input=raw_input,
        normalized_text=prepared.normalized_text,
        params=prepared.params,
        decode_passes=settings.default_decode_passes,
        threshold=resolved_threshold,
        persist=persist,
        metadata=metadata,
    )
