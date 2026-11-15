"""nas_status: read-only TrueNAS report and its JSON-RPC client."""

from __future__ import annotations

import json
import threading

import pytest

from openjarvis.tools.nas_status import (
    NasStatusTool,
    TrueNASClient,
    TrueNASError,
    build_report,
)

KEY = "1-secretkeyvalue"


@pytest.fixture(autouse=True)
def _register_tool():
    """Re-register after any registry clear (as other tool tests do)."""
    from openjarvis.core.registry import ToolRegistry

    if not ToolRegistry.contains("nas_status"):
        ToolRegistry.register_value("nas_status", NasStatusTool)


# Shapes as returned by TrueNAS 25.10.
RESPONSES = {
    "system.info": {
        "version": "25.10.4",
        "uptime_seconds": 93784,
        "physmem": 13601505280,
        "loadavg": [0.26, 0.4, 0.57],
    },
    "alert.list": [
        {
            "level": "INFO",
            "klass": "AppUpdate",
            "dismissed": False,
            "formatted": (
                "Updates are available for 3 applications: immich, tailscale, pihole"
            ),
        },
        {
            "level": "CRITICAL",
            "klass": "CertificateExpired",
            "dismissed": False,
            "formatted": "Certificate 'nextcloud-tailscale' has expired.",
        },
        {
            "level": "WARNING",
            "klass": "Old",
            "dismissed": True,
            "formatted": "dismissed one",
        },
    ],
    "pool.query": [
        {
            "name": "family",
            "status": "ONLINE",
            "healthy": True,
            "size": 1000 * 2**30,
            "allocated": 280 * 2**30,
            "scan": {
                "function": "SCRUB",
                "end_time": {"$date": 1792276578000},
                "errors": 0,
            },
        },
    ],
    "app.query": [
        {"name": "openjarvis", "state": "RUNNING"},
        {"name": "immich", "state": "RUNNING"},
        {"name": "pihole", "state": "STOPPED"},
    ],
    "pool.snapshottask.query": [
        {
            "dataset": "family",
            "lifetime_value": 2,
            "lifetime_unit": "WEEK",
            "state": {"state": "FINISHED", "datetime": {"$date": 1792276578000}},
        },
    ],
}


def fake_rpc(method, params=None):
    return RESPONSES[method]


# -- report -------------------------------------------------------------------


def test_report_covers_everything():
    report = build_report(fake_rpc)
    assert "SYSTEM: TrueNAS 25.10.4, up 1d 2h" in report
    assert "RAM 12.7 GiB" in report
    # Most severe first; dismissed alerts hidden.
    alerts = [line for line in report.splitlines() if line.startswith("- [")]
    assert alerts[0].startswith("- [CRITICAL] Certificate 'nextcloud-tailscale'")
    assert "dismissed one" not in report
    assert "POOL family: ONLINE, healthy, 28% used, last scrub" in report
    assert "0 errors" in report
    assert "APPS: openjarvis RUNNING, immich RUNNING, pihole STOPPED" in report
    assert (
        "SNAPSHOTS family: last run" in report and "(FINISHED), keep 2 week" in report
    )


def test_report_without_alerts_or_scrub():
    responses = dict(RESPONSES)
    responses["alert.list"] = []
    responses["pool.query"] = [
        {"name": "family", "status": "DEGRADED", "healthy": False, "scan": None}
    ]
    report = build_report(lambda m, p=None: responses[m])
    assert "ALERTS: none" in report
    assert "POOL family: DEGRADED, NOT healthy, usage ?, no scrub recorded" in report


# -- tool -----------------------------------------------------------------------


class FakeClient:
    def __init__(self, url, key, *, verify_ssl, timeout):
        self.args = (url, key, verify_ssl, timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def call(self, method, params=None):
        return fake_rpc(method, params)


def _config(**kw):
    from openjarvis.core.config import TrueNasToolConfig

    return TrueNasToolConfig(**{"url": "wss://nas/api/current", **kw})


def test_tool_reports(monkeypatch):
    monkeypatch.setenv("TRUENAS_API_KEY", KEY)
    made = []

    def factory(*a, **kw):
        made.append(FakeClient(*a, **kw))
        return made[-1]

    result = NasStatusTool(config=_config(), client_factory=factory).execute()
    assert result.success
    assert "POOL family: ONLINE" in result.content
    assert made[0].args == ("wss://nas/api/current", KEY, False, 15.0)


def test_tool_needs_url_and_key(monkeypatch):
    monkeypatch.delenv("TRUENAS_API_KEY", raising=False)
    r = NasStatusTool(config=_config(url="")).execute()
    assert not r.success and "not configured" in r.content
    r = NasStatusTool(config=_config(), client_factory=FakeClient).execute()
    assert not r.success and "TRUENAS_API_KEY" in r.content


def test_unreachable_nas_is_reported_without_the_key(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionRefusedError(f"cannot connect with {KEY}")

    r = NasStatusTool(api_key=KEY, config=_config(), client_factory=boom).execute()
    assert not r.success
    assert "ConnectionRefusedError" in r.content and "may be down" in r.content
    assert KEY not in r.content


def test_rejected_key_message(monkeypatch):
    def rejected(*a, **kw):
        raise TrueNASError("API key rejected")

    r = NasStatusTool(api_key=KEY, config=_config(), client_factory=rejected).execute()
    assert "API key rejected" in r.content and KEY not in r.content


def test_registered_and_spec():
    from openjarvis.core.registry import ToolRegistry

    assert ToolRegistry.contains("nas_status")
    spec = NasStatusTool(config=_config()).spec
    assert spec.name == "nas_status" and "Read-only" in spec.description


def test_config_from_toml(tmp_path):
    from openjarvis.core.config import load_config

    path = tmp_path / "config.toml"
    path.write_text(
        '[tools.truenas]\nurl = "wss://192.168.1.124/api/current"\n', encoding="utf-8"
    )
    cfg = load_config(path).tools.truenas
    assert (cfg.url, cfg.verify_ssl) == ("wss://192.168.1.124/api/current", False)


# -- JSON-RPC client against a local WebSocket server ---------------------------


@pytest.fixture
def server():
    from websockets.sync.server import serve

    calls = []

    def handler(ws):
        for raw in ws:
            msg = json.loads(raw)
            calls.append(msg["method"])
            if msg["method"] == "auth.login_with_api_key":
                ok = msg["params"] == [KEY]
                ws.send(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": ok}))
            elif msg["method"] == "boom":
                ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "error": {
                                "code": -32001,
                                "message": "Method call error",
                                "data": {"reason": "[ENOMETHOD] nope"},
                            },
                        }
                    )
                )
            else:
                # An unrelated notification arrives before the answer.
                ws.send(
                    json.dumps(
                        {"jsonrpc": "2.0", "method": "collection_update", "params": {}}
                    )
                )
                ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "result": RESPONSES[msg["method"]],
                        }
                    )
                )

    srv = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = srv.socket.getsockname()[1]
    yield f"ws://127.0.0.1:{port}", calls
    srv.shutdown()


def test_client_logs_in_and_skips_notifications(server):
    url, calls = server
    with TrueNASClient(url, KEY, verify_ssl=False, timeout=5) as client:
        assert client.call("system.info")["version"] == "25.10.4"
        report = build_report(client.call)
    assert calls[0] == "auth.login_with_api_key"
    assert "POOL family" in report


def test_client_rejects_bad_key(server):
    url, _ = server
    with pytest.raises(TrueNASError, match="rejected"):
        with TrueNASClient(url, "wrong", verify_ssl=False, timeout=5):
            pass


def test_client_surfaces_rpc_errors(server):
    url, _ = server
    with TrueNASClient(url, KEY, verify_ssl=False, timeout=5) as client:
        with pytest.raises(TrueNASError, match="ENOMETHOD"):
            client.call("boom")
