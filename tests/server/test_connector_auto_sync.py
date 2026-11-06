"""Periodic re-sync of connected connectors (server.connector_sync_minutes)."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from openjarvis.server.connectors_router import (  # noqa: E402
    _instances,
    create_connectors_router,
)


class _Connector:
    def __init__(self, connected: bool) -> None:
        self._connected = connected

    def is_connected(self) -> bool:
        return self._connected


@pytest.fixture
def router():
    r = create_connectors_router()
    yield r
    r.stop_auto_sync()


@pytest.fixture
def two_connectors(monkeypatch):
    from openjarvis.core.registry import ConnectorRegistry

    monkeypatch.setattr(
        "openjarvis.server.connectors_router._ensure_connectors_registered",
        lambda: None,
    )
    monkeypatch.setattr(ConnectorRegistry, "keys", classmethod(lambda cls: ["a", "b"]))
    saved = dict(_instances)
    _instances.clear()
    _instances["a"] = _Connector(connected=True)
    _instances["b"] = _Connector(connected=False)
    yield
    _instances.clear()
    _instances.update(saved)


def test_auto_sync_once_starts_only_connected(router, two_connectors):
    synced = []
    done = threading.Event()

    def fake_sync(self, connector, **_kw):
        synced.append(connector)
        done.set()
        return 0

    # Wait inside the patch: the sync runs on a worker thread.
    with patch("openjarvis.connectors.sync_engine.SyncEngine.sync", fake_sync):
        started = router.auto_sync_once()
        assert done.wait(5)
    assert started == ["a"]
    assert synced == [_instances["a"]]


def test_auto_sync_skips_a_connector_already_syncing(router, two_connectors):
    release = threading.Event()

    def slow_sync(self, connector, **_kw):
        release.wait(5)
        return 0

    with patch("openjarvis.connectors.sync_engine.SyncEngine.sync", slow_sync):
        assert router.auto_sync_once() == ["a"]
        assert router.auto_sync_once() == []
        release.set()


def test_auto_sync_errors_do_not_stop_the_loop(router, two_connectors, caplog):
    _instances["a"] = SimpleNamespace(
        is_connected=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert router.auto_sync_once() == []
    assert "Auto-sync of a did not start" in caplog.text


def test_start_auto_sync_disabled_for_non_positive_interval(router):
    assert router.start_auto_sync(0) is None
    assert router.start_auto_sync(-5) is None


def test_start_auto_sync_ticks_and_stops(router):
    # The loop calls the real auto_sync_once, whose first step is
    # _ensure_connectors_registered: use that to observe a tick.
    calls = []
    with (
        patch(
            "openjarvis.server.connectors_router._ensure_connectors_registered",
            lambda: calls.append("tick"),
        ),
        patch(
            "openjarvis.core.registry.ConnectorRegistry.keys",
            classmethod(lambda cls: []),
        ),
    ):
        thread = router.start_auto_sync(0.05)
        assert thread is not None and thread.name == "connector-auto-sync"
        for _ in range(100):
            if calls:
                break
            threading.Event().wait(0.02)
        router.stop_auto_sync()
        thread.join(2)
    assert "tick" in calls
    assert not thread.is_alive()


# -- app wiring ----------------------------------------------------------------


def _config(minutes):
    return SimpleNamespace(server=SimpleNamespace(connector_sync_minutes=minutes))


@pytest.mark.parametrize("config", [None, "zero"])
def test_app_does_not_start_auto_sync_without_explicit_positive_config(config):
    from fastapi.testclient import TestClient

    import openjarvis.server.app as app_mod
    from openjarvis.server.app import create_app

    real_factory = app_mod.create_connectors_router
    created = []

    def factory():
        r = real_factory()
        r.start_auto_sync = lambda s: created.append(s)
        return r

    cfg = None if config is None else _config(0)
    with patch.object(app_mod, "create_connectors_router", factory):
        try:
            app = create_app(engine=None, model="m", config=cfg)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            pytest.skip(f"create_app unavailable here: {exc}")
        with TestClient(app):
            pass
    assert created == []


def test_app_starts_auto_sync_with_configured_minutes():
    from fastapi.testclient import TestClient

    import openjarvis.server.app as app_mod
    from openjarvis.core.config import JarvisConfig
    from openjarvis.server.app import create_app

    real_factory = app_mod.create_connectors_router
    started, stopped = [], []

    def factory():
        r = real_factory()
        r.start_auto_sync = lambda s: started.append(s)
        r.stop_auto_sync = lambda: stopped.append(True)
        return r

    cfg = JarvisConfig()
    cfg.server.connector_sync_minutes = 15
    with patch.object(app_mod, "create_connectors_router", factory):
        try:
            app = create_app(engine=None, model="m", config=cfg)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            pytest.skip(f"create_app unavailable here: {exc}")
        with TestClient(app):
            assert started == [900.0]
    assert stopped == [True]


def test_config_default_is_off_and_toml_overrides(tmp_path):
    from openjarvis.core.config import JarvisConfig, load_config

    assert JarvisConfig().server.connector_sync_minutes == 0
    path = tmp_path / "config.toml"
    path.write_text("[server]\nconnector_sync_minutes = 15\n")
    assert load_config(path).server.connector_sync_minutes == 15
