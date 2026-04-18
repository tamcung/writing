from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_elasticsearch_client():
    settings = get_settings()
    if not settings.elasticsearch_url:
        return None
    try:
        from elasticsearch import Elasticsearch
    except Exception:
        return None
    return Elasticsearch(settings.elasticsearch_url)
