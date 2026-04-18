from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import parse_qsl, urlsplit

from experiments.prepare_modsec_decoded_dataset import decode_query_text


@dataclass(slots=True)
class ParameterMatch:
    name: str
    source: str
    raw_value: str
    decoded_value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "source": self.source,
            "raw_value": self.raw_value,
            "decoded_value": self.decoded_value,
        }


@dataclass(slots=True)
class PreprocessedRequest:
    normalized_text: str
    params: list[ParameterMatch]


def _extract_pairs(query: str, source: str, decode_passes: int) -> list[ParameterMatch]:
    values: list[ParameterMatch] = []
    for key, raw_value in parse_qsl(query, keep_blank_values=True, strict_parsing=False):
        decoded_value = decode_query_text(raw_value, decode_passes)
        if not decoded_value:
            continue
        values.append(ParameterMatch(name=key or "<EMPTY>", source=source, raw_value=raw_value, decoded_value=decoded_value))
    return values


def _join_values(values: Iterable[str]) -> str:
    return " ".join(value for value in values if value).strip()


def preprocess_raw_input(raw_input: str, decode_passes: int = 1) -> PreprocessedRequest:
    raw = raw_input.strip()
    if not raw:
        return PreprocessedRequest(normalized_text="", params=[])

    params: list[ParameterMatch] = []
    first_line = raw.splitlines()[0] if raw.splitlines() else raw
    looks_like_http = "HTTP/" in first_line and ("GET " in first_line or "POST " in first_line or "PUT " in first_line)

    if looks_like_http:
        head, _, body = raw.partition("\n\n")
        if not body and "\r\n\r\n" in raw:
            head, _, body = raw.partition("\r\n\r\n")
        lines = [line.strip() for line in head.splitlines() if line.strip()]
        request_line = lines[0] if lines else ""
        parts = request_line.split()
        target = parts[1] if len(parts) >= 2 else ""
        split = urlsplit(target)
        if split.query:
            params.extend(_extract_pairs(split.query, "query", decode_passes))
        content_type = ""
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == "content-type":
                content_type = value.strip().lower()
        if body and ("application/x-www-form-urlencoded" in content_type or "=" in body):
            params.extend(_extract_pairs(body.strip(), "body", decode_passes))
    else:
        split = urlsplit(raw)
        if split.scheme and split.query:
            params.extend(_extract_pairs(split.query, "url", decode_passes))
        elif "=" in raw and "&" in raw or raw.startswith("?"):
            params.extend(_extract_pairs(raw.lstrip("?"), "query", decode_passes))

    if params:
        normalized = _join_values(param.decoded_value for param in params)
        return PreprocessedRequest(normalized_text=normalized, params=params)

    return PreprocessedRequest(
        normalized_text=raw,
        params=[ParameterMatch(name="payload", source="raw", raw_value=raw, decoded_value=raw)],
    )


def risk_level(probability: float, threshold: float) -> str:
    if probability >= max(0.85, threshold):
        return "high"
    if probability >= max(0.65, threshold):
        return "medium"
    if probability >= threshold:
        return "low"
    return "benign"
