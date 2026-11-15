"""Read-only TrueNAS status tool: "how is the NAS doing?".

Talks to the TrueNAS SCALE JSON-RPC 2.0 API over WebSocket
(``wss://<host>/api/current``) with an API key. Use a key that belongs to a
user whose only role is ``READONLY_ADMIN``: the tool only calls query methods,
and a read-only account guarantees nothing else is possible even if the model
misbehaves.

Configuration (``[tools.truenas]``): ``url`` and ``verify_ssl`` (TrueNAS ships
a self-signed certificate). The key comes from the tool credential store or the
``TRUENAS_API_KEY`` environment variable and never appears in output or errors.
"""

from __future__ import annotations

import itertools
import json
import ssl
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from openjarvis.core.credentials import get_tool_credential
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_TOOL_NAME = "nas_status"
_API_KEY_ENV = "TRUENAS_API_KEY"


class TrueNASError(RuntimeError):
    """A JSON-RPC error returned by TrueNAS (message only, never the key)."""


class TrueNASClient:
    """Minimal synchronous JSON-RPC 2.0 client for the TrueNAS WebSocket API."""

    def __init__(self, url: str, api_key: str, *, verify_ssl: bool, timeout: float):
        self._url = url
        self._api_key = api_key
        self._verify = verify_ssl
        self._timeout = timeout
        self._ws: Any = None
        self._ids = itertools.count(1)

    def __enter__(self) -> "TrueNASClient":
        from websockets.sync.client import connect

        ctx = None
        if self._url.startswith("wss://"):
            ctx = ssl.create_default_context()
            if not self._verify:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
        self._ws = connect(self._url, ssl=ctx, open_timeout=self._timeout)
        if self.call("auth.login_with_api_key", [self._api_key]) is not True:
            raise TrueNASError("API key rejected")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._ws is not None:
            self._ws.close()

    def call(self, method: str, params: Optional[list] = None) -> Any:
        msg_id = next(self._ids)
        self._ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "method": method,
                    "params": params or [],
                }
            )
        )
        while True:
            reply = json.loads(self._ws.recv(timeout=self._timeout))
            if reply.get("id") != msg_id:
                continue  # event notifications
            if "error" in reply:
                err = reply["error"] or {}
                data = err.get("data") or {}
                raise TrueNASError(
                    str(data.get("reason") or err.get("message") or "error")
                )
            return reply.get("result")


def _when(value: Any) -> str:
    """TrueNAS dates come as {"$date": epoch_ms}; render local, minute precision."""
    if isinstance(value, dict) and "$date" in value:
        value = value["$date"] / 1000
    if isinstance(value, (int, float)):
        return (
            datetime.fromtimestamp(value, tz=timezone.utc)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
        )
    if isinstance(value, str) and value:
        return value[:16]
    return "?"


def _gib(n: Any) -> str:
    return f"{n / 2**30:.1f} GiB" if isinstance(n, (int, float)) else "?"


def _uptime(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "?"
    days, rest = divmod(int(seconds), 86400)
    return (
        f"{days}d {rest // 3600}h" if days else f"{rest // 3600}h {rest % 3600 // 60}m"
    )


def build_report(rpc: Callable[[str, Optional[list]], Any]) -> str:
    """Collect status with read-only calls and render a compact report."""
    lines: List[str] = []

    info = rpc("system.info", None) or {}
    load = ", ".join(f"{x:.2f}" for x in (info.get("loadavg") or [])) or "?"
    lines.append(
        f"SYSTEM: TrueNAS {info.get('version', '?')}, "
        f"up {_uptime(info.get('uptime_seconds'))}, load {load}, "
        f"RAM {_gib(info.get('physmem'))}"
    )

    alerts = [a for a in (rpc("alert.list", None) or []) if not a.get("dismissed")]
    order = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "NOTICE": 3, "INFO": 4}
    alerts.sort(key=lambda a: order.get(str(a.get("level")), 9))
    if alerts:
        lines.append(f"ALERTS ({len(alerts)}):")
        for a in alerts[:10]:
            text = " ".join(str(a.get("formatted") or a.get("klass") or "").split())
            lines.append(f"- [{a.get('level', '?')}] {text[:200]}")
    else:
        lines.append("ALERTS: none")

    for pool in rpc("pool.query", None) or []:
        scan = pool.get("scan") or {}
        size, alloc = pool.get("size"), pool.get("allocated")
        used = (
            f"{alloc / size * 100:.0f}% used"
            if size and alloc is not None
            else "usage ?"
        )
        last_scrub = (
            f"last scrub {_when(scan.get('end_time'))}, "
            f"{scan.get('errors', '?')} errors"
            if scan.get("function") == "SCRUB" or scan.get("end_time")
            else "no scrub recorded"
        )
        health = "healthy" if pool.get("healthy") else "NOT healthy"
        lines.append(
            f"POOL {pool.get('name')}: {pool.get('status')}, {health}, "
            f"{used}, {last_scrub}"
        )

    apps = rpc("app.query", None) or []
    if apps:
        lines.append(
            "APPS: " + ", ".join(f"{a.get('name')} {a.get('state')}" for a in apps)
        )

    for task in rpc("pool.snapshottask.query", None) or []:
        state = task.get("state") or {}
        lines.append(
            f"SNAPSHOTS {task.get('dataset')}: last run {_when(state.get('datetime'))} "
            f"({state.get('state', '?')}), keep {task.get('lifetime_value')} "
            f"{str(task.get('lifetime_unit', '')).lower()}"
        )
    return "\n".join(lines)


@ToolRegistry.register(_TOOL_NAME)
class NasStatusTool(BaseTool):
    """Report TrueNAS health: alerts, pool, apps, snapshots, uptime."""

    tool_id = _TOOL_NAME
    is_local = False

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        config: Any = None,
        client_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if config is None:
            try:
                from openjarvis.core.config import load_config

                config = load_config().tools.truenas
            except Exception:
                config = None
        self._url = str(getattr(config, "url", "") or "")
        self._verify = bool(getattr(config, "verify_ssl", False))
        self._timeout = float(getattr(config, "timeout", 15) or 15)
        self._api_key = (api_key or "").strip() or None
        self._client_factory = client_factory or TrueNASClient

    def _key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        try:
            key = get_tool_credential(_TOOL_NAME, _API_KEY_ENV)
        except (AttributeError, OSError, TypeError, ValueError):
            key = None
        return key.strip() if isinstance(key, str) and key.strip() else None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=_TOOL_NAME,
            description=(
                "Read-only health report of the user's NAS (TrueNAS): active "
                "alerts, storage pool health and usage, last scrub, app states, "
                "last automatic snapshot, uptime and load. Use it whenever the "
                "user asks how the NAS / server / disks / backups are doing."
            ),
            parameters={"type": "object", "properties": {}},
            category="system",
        )

    def execute(self, **params: Any) -> ToolResult:
        if not self._url:
            return ToolResult(
                tool_name=_TOOL_NAME,
                content="NAS status is not configured ([tools.truenas] url).",
                success=False,
            )
        key = self._key()
        if not key:
            return ToolResult(
                tool_name=_TOOL_NAME,
                content=(
                    f"NAS status needs {_API_KEY_ENV} (a read-only TrueNAS API key)."
                ),
                success=False,
            )
        try:
            with self._client_factory(
                self._url, key, verify_ssl=self._verify, timeout=self._timeout
            ) as client:
                report = build_report(client.call)
        except Exception as exc:  # noqa: BLE001 - report reachability, not secrets
            reason = str(exc) if isinstance(exc, TrueNASError) else type(exc).__name__
            return ToolResult(
                tool_name=_TOOL_NAME,
                content=f"Could not get NAS status from {self._url}: {reason}. "
                "If the NAS does not answer at all, it may be down.",
                success=False,
            )
        return ToolResult(tool_name=_TOOL_NAME, content=report, success=True)


__all__ = ["NasStatusTool", "TrueNASClient", "TrueNASError", "build_report"]
