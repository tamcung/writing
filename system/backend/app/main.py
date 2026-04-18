from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.router import router
from app.core.config import get_settings
from app.core.database import Base, engine
from app.services.model_manager import get_model_manager


settings = get_settings()
Base.metadata.create_all(bind=engine)


def _ensure_schema_updates() -> None:
    inspector = inspect(engine)
    detection_columns = {column["name"] for column in inspector.get_columns("detection_records")}
    if "timing_ms" not in detection_columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE detection_records ADD COLUMN timing_ms FLOAT DEFAULT 0"))


_ensure_schema_updates()

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix=settings.api_prefix)


@app.on_event("startup")
def warmup_services() -> None:
    get_model_manager()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "SQLi detection system backend is running."}
