"""Open-Meteo weather connector — no account, no API key, metric units.

The bundled ``weather`` connector needs an OpenWeatherMap key and reports
Fahrenheit. Open-Meteo is free for non-commercial use without registration, so
a morning digest can include the weather with zero setup beyond a place name.

Configuration (no secrets) lives in ``~/.openjarvis/connectors/open_meteo.json``
and is written by :meth:`OpenMeteoConnector.configure`, which resolves the
place name with Open-Meteo's geocoding API. All network calls are module-level
functions so tests can mock them.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import httpx

from openjarvis.connectors._stubs import BaseConnector, Document, SyncStatus
from openjarvis.core.config import DEFAULT_CONFIG_DIR
from openjarvis.core.registry import ConnectorRegistry

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "connectors" / "open_meteo.json"

# WMO weather interpretation codes (Open-Meteo docs), in Spanish and English.
_WMO = {
    0: ("despejado", "clear sky"),
    1: ("mayormente despejado", "mainly clear"),
    2: ("parcialmente nublado", "partly cloudy"),
    3: ("cubierto", "overcast"),
    45: ("niebla", "fog"),
    48: ("niebla con escarcha", "depositing rime fog"),
    51: ("llovizna débil", "light drizzle"),
    53: ("llovizna", "drizzle"),
    55: ("llovizna intensa", "dense drizzle"),
    56: ("llovizna helada", "freezing drizzle"),
    57: ("llovizna helada intensa", "dense freezing drizzle"),
    61: ("lluvia débil", "slight rain"),
    63: ("lluvia", "rain"),
    65: ("lluvia fuerte", "heavy rain"),
    66: ("lluvia helada", "freezing rain"),
    67: ("lluvia helada fuerte", "heavy freezing rain"),
    71: ("nieve débil", "slight snow"),
    73: ("nieve", "snow"),
    75: ("nieve fuerte", "heavy snow"),
    77: ("granos de nieve", "snow grains"),
    80: ("chubascos débiles", "slight showers"),
    81: ("chubascos", "showers"),
    82: ("chubascos fuertes", "violent showers"),
    85: ("chubascos de nieve", "snow showers"),
    86: ("chubascos de nieve fuertes", "heavy snow showers"),
    95: ("tormenta", "thunderstorm"),
    96: ("tormenta con granizo", "thunderstorm with hail"),
    99: ("tormenta con granizo fuerte", "thunderstorm with heavy hail"),
}


def describe_weather_code(code: Any, language: str = "es") -> str:
    """Human description of a WMO weather code (unknown codes stay numeric)."""
    try:
        es, en = _WMO[int(code)]
    except (KeyError, TypeError, ValueError):
        return f"código {code}" if language == "es" else f"code {code}"
    return es if language == "es" else en


def _geocode(name: str, language: str) -> Dict[str, Any]:
    resp = httpx.get(
        _GEOCODE_URL,
        params={"name": name, "count": 1, "language": language, "format": "json"},
        timeout=20.0,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"Open-Meteo no encuentra el lugar {name!r}")
    return results[0]


def _forecast(latitude: float, longitude: float, timezone: str) -> Dict[str, Any]:
    resp = httpx.get(
        _FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone or "auto",
            "forecast_days": 1,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,precipitation_sum"
            ),
        },
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def format_forecast(place: str, data: Dict[str, Any], language: str = "es") -> str:
    """One-line summary of an Open-Meteo forecast response."""
    current = data.get("current") or {}
    daily = data.get("daily") or {}

    def first(key: str) -> Any:
        values = daily.get(key) or []
        return values[0] if values else None

    tmin, tmax = first("temperature_2m_min"), first("temperature_2m_max")
    rain_prob, rain_mm = (
        first("precipitation_probability_max"),
        first("precipitation_sum"),
    )
    day_desc = describe_weather_code(first("weather_code"), language)
    now_temp = current.get("temperature_2m")
    now_desc = describe_weather_code(current.get("weather_code"), language)
    wind = current.get("wind_speed_10m")

    def deg(v: Any) -> str:
        return f"{round(v)} °C" if isinstance(v, (int, float)) else "?"

    if language == "es":
        parts = [f"{place} hoy: {day_desc}, entre {deg(tmin)} y {deg(tmax)}"]
        if rain_prob is not None:
            extra = (
                f", {rain_mm:.1f} mm"
                if isinstance(rain_mm, (int, float)) and rain_mm
                else ""
            )
            parts.append(f"probabilidad de lluvia {rain_prob}%{extra}")
        if now_temp is not None:
            parts.append(f"ahora {deg(now_temp)} y {now_desc}")
        if isinstance(wind, (int, float)):
            parts.append(f"viento {round(wind)} km/h")
    else:
        parts = [f"{place} today: {day_desc}, {deg(tmin)} to {deg(tmax)}"]
        if rain_prob is not None:
            parts.append(f"chance of rain {rain_prob}%")
        if now_temp is not None:
            parts.append(f"now {deg(now_temp)}, {now_desc}")
        if isinstance(wind, (int, float)):
            parts.append(f"wind {round(wind)} km/h")
    return "; ".join(parts) + "."


@ConnectorRegistry.register("open_meteo")
class OpenMeteoConnector(BaseConnector):
    """Today's weather for one configured place, from Open-Meteo."""

    connector_id = "open_meteo"
    display_name = "Open-Meteo (weather)"
    auth_type = "local"

    def __init__(self, *, config_path: str | Path | None = None) -> None:
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._status = SyncStatus()

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self._config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def configure(self, location: str, *, language: str = "es") -> Dict[str, Any]:
        """Resolve *location* and save it. Returns the stored configuration."""
        place = _geocode(location, language)
        config = {
            "location": place.get("name") or location,
            "admin": place.get("admin1", ""),
            "country": place.get("country", ""),
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "timezone": place.get("timezone", ""),
            "language": language,
        }
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return config

    def is_connected(self) -> bool:
        cfg = self._load()
        return "latitude" in cfg and "longitude" in cfg

    def disconnect(self) -> None:
        try:
            self._config_path.unlink()
        except FileNotFoundError:
            pass

    def sync(
        self, *, since: Optional[datetime] = None, cursor: Optional[str] = None
    ) -> Iterator[Document]:
        cfg = self._load()
        if "latitude" not in cfg:
            return
        language = cfg.get("language", "es")
        data = _forecast(cfg["latitude"], cfg["longitude"], cfg.get("timezone", ""))
        place = cfg.get("location", "")
        # One stable document per day: re-syncs replace it instead of piling up.
        day = (data.get("daily") or {}).get(
            "time", [datetime.now().date().isoformat()]
        )[0]
        self._status.items_synced = 1
        yield Document(
            doc_id=f"open_meteo:{place}:{day}",
            source="open_meteo",
            doc_type="weather",
            content=format_forecast(place, data, language),
            title=f"{place} {day}",
            timestamp=datetime.now(),
            metadata={"latitude": cfg["latitude"], "longitude": cfg["longitude"]},
        )

    def sync_status(self) -> SyncStatus:
        return self._status


__all__ = ["OpenMeteoConnector", "describe_weather_code", "format_forecast"]
