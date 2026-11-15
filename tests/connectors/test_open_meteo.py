"""Open-Meteo weather connector (no key, metric) and its digest formatting."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from openjarvis.connectors.open_meteo import (
    OpenMeteoConnector,
    describe_weather_code,
    format_forecast,
)
from openjarvis.core.registry import ConnectorRegistry

_GEO = {
    "name": "Vigo",
    "admin1": "Galicia",
    "country": "España",
    "latitude": 42.2328,
    "longitude": -8.7226,
    "timezone": "Europe/Madrid",
}
_FORECAST = {
    "current": {"temperature_2m": 14.6, "weather_code": 61, "wind_speed_10m": 18.2},
    "daily": {
        "time": ["2026-11-16"],
        "weather_code": [63],
        "temperature_2m_max": [16.4],
        "temperature_2m_min": [11.2],
        "precipitation_probability_max": [85],
        "precipitation_sum": [7.3],
    },
}


@pytest.fixture(autouse=True)
def _registered():
    if not ConnectorRegistry.contains("open_meteo"):
        ConnectorRegistry.register_value("open_meteo", OpenMeteoConnector)


def test_codes_are_described_in_spanish_and_english():
    assert describe_weather_code(0) == "despejado"
    assert describe_weather_code(95, "en") == "thunderstorm"
    assert describe_weather_code(1234) == "código 1234"
    assert describe_weather_code(None) == "código None"


def test_spanish_summary_is_metric():
    line = format_forecast("Vigo", _FORECAST, "es")
    assert line == (
        "Vigo hoy: lluvia, entre 11 °C y 16 °C; probabilidad de lluvia 85%, "
        "7.3 mm; ahora 15 °C y lluvia débil; viento 18 km/h."
    )
    assert "°F" not in line


def test_missing_fields_degrade_gracefully():
    assert format_forecast("Vigo", {}, "es") == "Vigo hoy: código None, entre ? y ?."


def test_not_connected_until_configured(tmp_path):
    conn = OpenMeteoConnector(config_path=tmp_path / "om.json")
    assert not conn.is_connected()
    assert list(conn.sync()) == []


def test_configure_geocodes_and_stores_no_secrets(tmp_path):
    path = tmp_path / "om.json"
    conn = OpenMeteoConnector(config_path=path)
    with patch("openjarvis.connectors.open_meteo._geocode", return_value=_GEO) as geo:
        cfg = conn.configure("Vigo")
    geo.assert_called_once_with("Vigo", "es")
    assert conn.is_connected()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == cfg
    assert stored["location"] == "Vigo" and stored["timezone"] == "Europe/Madrid"


def test_sync_yields_one_stable_document_per_day(tmp_path):
    conn = OpenMeteoConnector(config_path=tmp_path / "om.json")
    with patch("openjarvis.connectors.open_meteo._geocode", return_value=_GEO):
        conn.configure("Vigo")
    with patch(
        "openjarvis.connectors.open_meteo._forecast", return_value=_FORECAST
    ) as fc:
        (doc,) = list(conn.sync())
    fc.assert_called_once_with(42.2328, -8.7226, "Europe/Madrid")
    assert doc.doc_id == "open_meteo:Vigo:2026-11-16"
    assert doc.source == "open_meteo" and doc.doc_type == "weather"
    assert doc.content.startswith("Vigo hoy: lluvia")


def test_digest_collect_puts_it_under_world(tmp_path, monkeypatch):
    from openjarvis.tools.digest_collect import DigestCollectTool

    conn = OpenMeteoConnector(config_path=tmp_path / "om.json")
    with patch("openjarvis.connectors.open_meteo._geocode", return_value=_GEO):
        conn.configure("Vigo")
    monkeypatch.setattr(
        ConnectorRegistry, "get", classmethod(lambda cls, key: lambda: conn)
    )
    with patch("openjarvis.connectors.open_meteo._forecast", return_value=_FORECAST):
        result = DigestCollectTool().execute(sources=["open_meteo"])
    assert "=== WORLD ===" in result.content
    assert "[weather] Vigo hoy: lluvia" in result.content
