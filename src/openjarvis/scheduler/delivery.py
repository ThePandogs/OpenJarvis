"""Deliver scheduled-task results to a chat channel, and keep digest tasks in sync.

A ``ScheduledTask`` whose ``metadata["deliver"]`` is
``{"channel": "telegram", "to": "<chat id>", "voice": true}`` has its result
pushed to that chat when it runs: the text first, then (optionally) a voice
note. Without this, scheduled runs only landed in ``scheduler.db`` and nobody
heard about them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DIGEST_KIND = "digest"


def _content_of(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("content", "") or "")
    return "" if result is None else str(result)


def _metadata_of(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        meta = result.get("metadata") or {}
        return meta if isinstance(meta, dict) else {}
    return {}


def _voice_from_wav(path: str) -> Optional[bytes]:
    """OGG/Opus from an agent-produced WAV, so it is not synthesized twice."""
    if not path or not path.lower().endswith(".wav"):
        return None
    try:
        from openjarvis.channels._voice import wav_to_ogg_opus

        return wav_to_ogg_opus(Path(path).read_bytes())
    except Exception:  # noqa: BLE001
        logger.debug("Could not reuse %s as a voice note", path, exc_info=True)
        return None


def make_channel_delivery(channel: Any, channel_name: str) -> Callable[..., None]:
    """Return an ``on_result`` callback for :class:`TaskScheduler`."""

    def deliver(task: Any, success: bool, result: Any) -> None:
        spec = (getattr(task, "metadata", None) or {}).get("deliver") or {}
        if spec.get("channel") != channel_name or not spec.get("to"):
            return
        to = str(spec["to"])
        label = spec.get("label") or getattr(task, "prompt", "")[:40]

        if not success:
            channel.send(to, f"⚠ {label}: {result}")
            return

        text = _content_of(result).strip()
        if not text:
            return
        channel.send(to, text)
        if not spec.get("voice"):
            return
        meta = _metadata_of(result)
        ogg = _voice_from_wav(str(meta.get("audio_path") or ""))
        if ogg is not None and hasattr(channel, "send_voice_audio"):
            channel.send_voice_audio(to, ogg)
        elif hasattr(channel, "send_voice"):
            channel.send_voice(to, str(meta.get("spoken_text") or text))

    return deliver


def digest_schedules(digest_config: Any) -> List[str]:
    schedules = [s for s in (getattr(digest_config, "schedules", None) or []) if s]
    if schedules:
        return [str(s).strip() for s in schedules]
    single = str(getattr(digest_config, "schedule", "") or "").strip()
    return [single] if single else []


def sync_digest_tasks(
    scheduler: Any,
    digest_config: Any,
    deliver: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Make the scheduler's digest tasks match ``[digest]`` exactly.

    Tasks are tagged ``metadata.kind == "digest"``. Any whose cron, timezone or
    delivery differs from the configuration is cancelled and recreated, and
    unmatched ones are cancelled, so editing config.toml and restarting is
    enough. Returns a description of the active digest tasks.
    """
    tz = str(getattr(digest_config, "timezone", "") or "")
    wanted = {
        cron: {"kind": DIGEST_KIND, "timezone": tz, "deliver": deliver or {}}
        for cron in digest_schedules(digest_config)
    }

    def key(meta: Dict[str, Any]) -> str:
        return json.dumps(
            {k: meta.get(k) for k in ("timezone", "deliver")}, sort_keys=True
        )

    kept: Dict[str, Any] = {}
    for task in scheduler.list_tasks(status="active"):
        meta = task.metadata or {}
        if meta.get("kind") != DIGEST_KIND:
            continue
        target = wanted.get(task.schedule_value)
        if (
            target is not None
            and task.schedule_value not in kept
            and key(meta) == key(target)
        ):
            kept[task.schedule_value] = task
        else:
            scheduler.cancel_task(task.id)

    for cron, meta in wanted.items():
        if cron not in kept:
            kept[cron] = scheduler.create_task(
                "Morning digest",
                "cron",
                cron,
                agent="morning_digest",
                metadata=meta,
            )
    return [f"{cron} ({tz or 'UTC'}) next {t.next_run}" for cron, t in kept.items()]


__all__ = [
    "DIGEST_KIND",
    "digest_schedules",
    "make_channel_delivery",
    "sync_digest_tasks",
]
