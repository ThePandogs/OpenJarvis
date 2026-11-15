"""Scheduled results reach the chat; digest tasks follow [digest]; times are UTC."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openjarvis.scheduler.delivery import (
    digest_schedules,
    make_channel_delivery,
    sync_digest_tasks,
)
from openjarvis.scheduler.scheduler import TaskScheduler, _once_to_utc
from openjarvis.scheduler.store import SchedulerStore


@pytest.fixture
def scheduler(tmp_path):
    store = SchedulerStore(db_path=str(tmp_path / "sched.db"))
    yield TaskScheduler(store)
    store.close()


def _task(deliver=None, prompt="Morning digest"):
    return SimpleNamespace(
        prompt=prompt, metadata={"deliver": deliver} if deliver else {}
    )


# -- one-shot times -----------------------------------------------------------


def test_naive_once_time_is_read_in_the_task_timezone():
    # 18:00 in Madrid in November (UTC+1) is 17:00 UTC, not 18:00 UTC.
    assert _once_to_utc("2026-11-20T18:00:00", "Europe/Madrid") == (
        "2026-11-20T17:00:00+00:00"
    )
    # Summer time: UTC+2.
    assert _once_to_utc("2026-07-20T18:00:00", "Europe/Madrid") == (
        "2026-07-20T16:00:00+00:00"
    )


def test_aware_once_time_is_only_converted():
    assert _once_to_utc("2026-11-20T18:00:00+01:00") == "2026-11-20T17:00:00+00:00"
    assert _once_to_utc("2026-11-20T17:00:00Z") == "2026-11-20T17:00:00+00:00"


def test_garbage_once_value_is_left_alone():
    assert _once_to_utc("tomorrow at six") == "tomorrow at six"


def test_create_once_task_stores_utc(scheduler):
    task = scheduler.create_task(
        "llamar al taller",
        "once",
        "2026-11-20T18:00:00",
        metadata={"timezone": "Europe/Madrid"},
    )
    assert task.next_run == "2026-11-20T17:00:00+00:00"


# -- execution with a real-shaped system result --------------------------------


def test_dict_result_is_logged_and_task_advances(scheduler):
    """JarvisSystem.ask returns a dict; it used to crash log_run before
    next_run was advanced, re-running the task on every poll."""
    seen = []
    system = MagicMock()
    system.ask.return_value = {"content": "hola", "metadata": {"spoken_text": "hola"}}
    sched = TaskScheduler(
        scheduler._store, system, on_result=lambda t, ok, r: seen.append((ok, r))
    )
    task = sched.create_task("x", "once", "2020-01-01T00:00:00+00:00")
    sched._execute_task(task)

    stored = sched._store.get_task(task.id)
    assert stored["status"] == "completed"
    assert sched._store.get_run_logs(task.id)[0]["result"] == "hola"
    assert seen == [(True, {"content": "hola", "metadata": {"spoken_text": "hola"}})]


def test_on_result_errors_do_not_break_bookkeeping(scheduler):
    def boom(*_a):
        raise RuntimeError("delivery down")

    sched = TaskScheduler(scheduler._store, None, on_result=boom)
    task = sched.create_task("x", "once", "2020-01-01T00:00:00+00:00")
    sched._execute_task(task)
    assert sched._store.get_task(task.id)["status"] == "completed"


# -- delivery -------------------------------------------------------------------


def _channel():
    ch = MagicMock()
    ch.send.return_value = True
    ch.send_voice.return_value = True
    ch.send_voice_audio.return_value = True
    return ch


def test_delivers_text_then_voice_from_spoken_text():
    ch = _channel()
    deliver = make_channel_delivery(ch, "telegram")
    result = {
        "content": "Buenos días, Carlos.\n\nNAS OK - comprobado 07:55",
        "metadata": {"spoken_text": "Buenos días, Carlos."},
    }
    deliver(_task({"channel": "telegram", "to": "555", "voice": True}), True, result)
    ch.send.assert_called_once_with(
        "555", "Buenos días, Carlos.\n\nNAS OK - comprobado 07:55"
    )
    # The status footer is shown, not spoken.
    ch.send_voice.assert_called_once_with("555", "Buenos días, Carlos.")


def test_reuses_agent_wav_instead_of_synthesizing(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("av")
    wav = tmp_path / "digest.wav"
    buf = io.BytesIO()
    sf.write(buf, np.zeros(2400, dtype="float32"), 24000, format="WAV")
    wav.write_bytes(buf.getvalue())

    ch = _channel()
    make_channel_delivery(ch, "telegram")(
        _task({"channel": "telegram", "to": "555", "voice": True}),
        True,
        {"content": "hola", "metadata": {"audio_path": str(wav)}},
    )
    ogg = ch.send_voice_audio.call_args.args[1]
    assert ogg[:4] == b"OggS"
    ch.send_voice.assert_not_called()


def test_text_only_when_voice_off():
    ch = _channel()
    make_channel_delivery(ch, "telegram")(
        _task({"channel": "telegram", "to": "555", "voice": False}), True, "hola"
    )
    ch.send.assert_called_once_with("555", "hola")
    ch.send_voice.assert_not_called()


def test_failure_is_reported_with_label():
    ch = _channel()
    make_channel_delivery(ch, "telegram")(
        _task({"channel": "telegram", "to": "555", "label": "Resumen de la mañana"}),
        False,
        "Ollama not reachable",
    )
    ch.send.assert_called_once_with(
        "555", "⚠ Resumen de la mañana: Ollama not reachable"
    )


@pytest.mark.parametrize(
    "deliver",
    [None, {"channel": "slack", "to": "555"}, {"channel": "telegram", "to": ""}],
)
def test_tasks_for_other_channels_or_without_target_are_ignored(deliver):
    ch = _channel()
    make_channel_delivery(ch, "telegram")(_task(deliver), True, "hola")
    ch.send.assert_not_called()


def test_empty_result_sends_nothing():
    ch = _channel()
    make_channel_delivery(ch, "telegram")(
        _task({"channel": "telegram", "to": "555", "voice": True}),
        True,
        {"content": " "},
    )
    ch.send.assert_not_called()


# -- digest task reconciliation -------------------------------------------------


def _digest_cfg(**kw):
    base = dict(schedule="0 6 * * *", schedules=[], timezone="Europe/Madrid")
    base.update(kw)
    return SimpleNamespace(**base)


def test_schedules_override_single_schedule():
    assert digest_schedules(_digest_cfg()) == ["0 6 * * *"]
    assert digest_schedules(_digest_cfg(schedules=["0 8 * * 1-5", "0 10 * * 0,6"])) == [
        "0 8 * * 1-5",
        "0 10 * * 0,6",
    ]


def test_sync_creates_one_task_per_schedule_in_local_time(scheduler):
    cfg = _digest_cfg(schedules=["0 8 * * 1-5", "0 10 * * 0,6"])
    deliver = {"channel": "telegram", "to": "555", "voice": True}
    sync_digest_tasks(scheduler, cfg, deliver)
    tasks = {t.schedule_value: t for t in scheduler.list_tasks(status="active")}
    assert set(tasks) == {"0 8 * * 1-5", "0 10 * * 0,6"}
    for t in tasks.values():
        assert t.agent == "morning_digest"
        assert t.metadata == {
            "kind": "digest",
            "timezone": "Europe/Madrid",
            "deliver": deliver,
        }
    # 08:00 Madrid on a weekday is 07:00 UTC in winter.
    weekday = datetime.fromisoformat(tasks["0 8 * * 1-5"].next_run)
    assert weekday.astimezone(timezone.utc).hour in (6, 7)  # 7 in winter, 6 in summer
    weekend = datetime.fromisoformat(tasks["0 10 * * 0,6"].next_run)
    assert weekend.weekday() in (5, 6)


def test_sync_is_idempotent_and_follows_config_changes(scheduler):
    deliver = {"channel": "telegram", "to": "555", "voice": True}
    cfg = _digest_cfg(schedules=["0 8 * * 1-5", "0 10 * * 0,6"])
    sync_digest_tasks(scheduler, cfg, deliver)
    first = {t.id for t in scheduler.list_tasks(status="active")}
    sync_digest_tasks(scheduler, cfg, deliver)
    assert {t.id for t in scheduler.list_tasks(status="active")} == first

    sync_digest_tasks(scheduler, _digest_cfg(schedules=["0 9 * * *"]), deliver)
    active = scheduler.list_tasks(status="active")
    assert [t.schedule_value for t in active] == ["0 9 * * *"]

    # Changing only the destination recreates the task.
    sync_digest_tasks(
        scheduler, _digest_cfg(schedules=["0 9 * * *"]), {**deliver, "to": "777"}
    )
    (task,) = scheduler.list_tasks(status="active")
    assert task.metadata["deliver"]["to"] == "777"


def test_sync_leaves_other_tasks_alone(scheduler):
    other = scheduler.create_task("recordatorio", "once", "2030-01-01T10:00:00+00:00")
    sync_digest_tasks(scheduler, _digest_cfg(schedule="", schedules=[]), {})
    assert [t.id for t in scheduler.list_tasks(status="active")] == [other.id]
