"""Task scheduler — cron/interval/once execution with background polling."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from openjarvis.scheduler.store import SchedulerStore

logger = logging.getLogger(__name__)

# Event type strings (avoids editing core EventType enum)
SCHEDULER_TASK_START = "scheduler_task_start"
SCHEDULER_TASK_END = "scheduler_task_end"


@dataclass(slots=True)
class ScheduledTask:
    """A task scheduled for future or recurring execution."""

    id: str
    prompt: str
    schedule_type: str  # "cron" | "interval" | "once"
    schedule_value: str  # cron expression, interval seconds, ISO datetime
    context_mode: str = "isolated"
    status: str = "active"
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    agent: str = "simple"
    tools: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for store persistence."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "schedule_type": self.schedule_type,
            "schedule_value": self.schedule_value,
            "context_mode": self.context_mode,
            "status": self.status,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "agent": self.agent,
            "tools": self.tools,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ScheduledTask:
        """Deserialize from a plain dict."""
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            schedule_type=d["schedule_type"],
            schedule_value=d["schedule_value"],
            context_mode=d.get("context_mode", "isolated"),
            status=d.get("status", "active"),
            next_run=d.get("next_run"),
            last_run=d.get("last_run"),
            agent=d.get("agent", "simple"),
            tools=d.get("tools", ""),
            metadata=d.get("metadata", {}),
        )


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _once_to_utc(value: str, timezone_name: str | None = None) -> str:
    """Normalise a one-shot target time to an aware UTC ISO string.

    Due tasks are found by comparing ``next_run`` with ``_now_iso()`` (UTC) as
    strings, so a naive "2026-11-20T18:00:00" meant as local time used to fire
    at 18:00 UTC -- an hour early in Madrid in winter, two in summer. A naive
    value is read in the task's ``timezone`` (or the machine's local zone) and
    converted; an aware one is just converted. Unparseable values are returned
    unchanged so existing behaviour does not break.
    """
    try:
        target = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return value
    if target.tzinfo is None:
        if timezone_name:
            target = target.replace(tzinfo=ZoneInfo(timezone_name))
        else:
            target = target.astimezone()  # naive -> local zone
    return target.astimezone(timezone.utc).isoformat()


class TaskScheduler:
    """Scheduler that polls for due tasks and executes them.

    Parameters
    ----------
    store:
        The persistence backend.
    system:
        Optional ``JarvisSystem`` instance for executing prompts.
    poll_interval:
        Seconds between poll cycles (default 60).
    bus:
        Optional event bus for publishing scheduler events.
    on_result:
        Optional ``callback(task, success, result)`` run after each execution,
        e.g. to deliver the result to a chat channel. Its errors are logged
        and never affect the task's own bookkeeping.
    """

    def __init__(
        self,
        store: SchedulerStore,
        system: Any = None,
        *,
        poll_interval: int = 60,
        bus: Any = None,
        on_result: Any = None,
    ) -> None:
        self._store = store
        self._system = system
        self._poll_interval = poll_interval
        self._bus = bus
        self._on_result = on_result
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # -- Public API ----------------------------------------------------------

    def start(self) -> None:
        """Start the background polling daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="jarvis-scheduler"
        )
        self._thread.start()
        logger.info("Scheduler started (poll_interval=%ds)", self._poll_interval)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 5)
            self._thread = None
        logger.info("Scheduler stopped")

    def create_task(
        self,
        prompt: str,
        schedule_type: str,
        schedule_value: str,
        **kwargs: Any,
    ) -> ScheduledTask:
        """Create and persist a new scheduled task."""
        task = ScheduledTask(
            id=uuid.uuid4().hex[:16],
            prompt=prompt,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            agent=kwargs.get("agent", "simple"),
            tools=kwargs.get("tools", ""),
            context_mode=kwargs.get("context_mode", "isolated"),
            metadata=kwargs.get("metadata", {}),
        )
        task.next_run = self._compute_next_run(task)
        with self._lock:
            self._store.save_task(task.to_dict())
        return task

    def list_tasks(self, *, status: Optional[str] = None) -> List[ScheduledTask]:
        """Return tasks, optionally filtered by *status*."""
        with self._lock:
            rows = self._store.list_tasks(status=status)
        return [ScheduledTask.from_dict(r) for r in rows]

    def pause_task(self, task_id: str) -> None:
        """Pause an active task."""
        with self._lock:
            d = self._store.get_task(task_id)
            if d is None:
                raise KeyError(f"Task not found: {task_id}")
            d["status"] = "paused"
            self._store.update_task(d)

    def resume_task(self, task_id: str) -> None:
        """Resume a paused task."""
        with self._lock:
            d = self._store.get_task(task_id)
            if d is None:
                raise KeyError(f"Task not found: {task_id}")
            d["status"] = "active"
            # Recompute next_run from now
            task = ScheduledTask.from_dict(d)
            task.next_run = self._compute_next_run(task)
            self._store.update_task(task.to_dict())

    def cancel_task(self, task_id: str) -> None:
        """Cancel a task (sets status to cancelled)."""
        with self._lock:
            d = self._store.get_task(task_id)
            if d is None:
                raise KeyError(f"Task not found: {task_id}")
            d["status"] = "cancelled"
            d["next_run"] = None
            self._store.update_task(d)

    # -- Background loop -----------------------------------------------------

    def _poll_loop(self) -> None:
        """Poll for due tasks and execute them until stopped."""
        while not self._stop_event.is_set():
            try:
                now = _now_iso()
                with self._lock:
                    due = self._store.get_due_tasks(now)
                for task_dict in due:
                    task = ScheduledTask.from_dict(task_dict)
                    self._execute_task(task)
            except Exception:
                logger.exception("Scheduler poll error")
            self._stop_event.wait(timeout=self._poll_interval)

    def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a single due task and log the result."""
        started_at = _now_iso()

        # Publish start event
        if self._bus is not None:
            self._bus.publish(
                SCHEDULER_TASK_START,
                {"task_id": task.id, "prompt": task.prompt},
            )

        success = False
        result_text = ""
        raw_result: Any = None
        error_text = ""

        try:
            if self._system is not None:
                raw_tools = (
                    task.tools
                    if isinstance(task.tools, list)
                    else task.tools.split(",")
                )
                tools_list = (
                    [t.strip() for t in raw_tools if t.strip()] if task.tools else []
                )
                ask_kwargs: Dict[str, Any] = {
                    "agent": task.agent,
                    "tools": tools_list if tools_list else None,
                }
                meta = task.metadata or {}
                if meta.get("operator_id"):
                    ask_kwargs["system_prompt"] = meta.get("system_prompt", "")
                    ask_kwargs["operator_id"] = meta["operator_id"]
                raw_result = self._system.ask(
                    task.prompt,
                    **ask_kwargs,
                )
                # JarvisSystem.ask returns a dict. Storing it as-is made
                # log_run raise (SQLite cannot bind a dict) BEFORE next_run was
                # advanced, so the task re-ran on every poll.
                result_text = (
                    str(raw_result.get("content", ""))
                    if isinstance(raw_result, dict)
                    else str(raw_result)
                )
            else:
                result_text = f"[dry-run] Would execute: {task.prompt}"
                raw_result = result_text
            success = True
        except Exception as exc:
            error_text = str(exc)
            logger.error("Task %s failed: %s", task.id, exc)

        finished_at = _now_iso()

        # Log the run
        with self._lock:
            self._store.log_run(
                task_id=task.id,
                started_at=started_at,
                finished_at=finished_at,
                success=success,
                result=result_text,
                error=error_text,
            )

            # Update task state
            d = self._store.get_task(task.id)
            if d is not None:
                d["last_run"] = finished_at
                next_run = self._compute_next_run(ScheduledTask.from_dict(d))
                d["next_run"] = next_run
                if next_run is None:
                    d["status"] = "completed"
                self._store.update_task(d)

        if self._on_result is not None:
            try:
                self._on_result(task, success, raw_result if success else error_text)
            except Exception:
                logger.exception("Scheduler on_result callback failed for %s", task.id)

        # Publish end event
        if self._bus is not None:
            self._bus.publish(
                SCHEDULER_TASK_END,
                {
                    "task_id": task.id,
                    "success": success,
                    "result": result_text,
                    "error": error_text,
                },
            )

    def _compute_next_run(self, task: ScheduledTask) -> Optional[str]:
        """Compute the next run time for a task.

        Returns an ISO 8601 string, or ``None`` if the task should not run again.
        """
        now = datetime.now(timezone.utc)

        if task.schedule_type == "once":
            # If already run, no more runs
            if task.last_run is not None:
                return None
            # Otherwise the schedule_value is the target ISO datetime
            timezone_name = str((task.metadata or {}).get("timezone") or "") or None
            return _once_to_utc(task.schedule_value, timezone_name)

        if task.schedule_type == "interval":
            seconds = float(task.schedule_value)
            next_time = now + timedelta(seconds=seconds)
            return next_time.isoformat()

        if task.schedule_type == "cron":
            timezone_name = str((task.metadata or {}).get("timezone") or "") or None
            return self._compute_next_cron(task.schedule_value, now, timezone_name)

        return None

    @staticmethod
    def _compute_next_cron(
        cron_expr: str,
        now: datetime,
        timezone_name: str | None = None,
    ) -> Optional[str]:
        """Compute the next run time from a cron expression.

        Evaluate in the configured IANA timezone and persist the result in UTC.
        Missing cron support is fatal instead of silently changing semantics.
        """
        try:
            from croniter import croniter  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "croniter is required for cron schedules; reinstall OpenJarvis"
            ) from exc

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        local_now = now.astimezone(ZoneInfo(timezone_name)) if timezone_name else now
        next_local = croniter(cron_expr, local_now).get_next(datetime)
        if next_local.tzinfo is None:
            next_local = next_local.replace(tzinfo=local_now.tzinfo)
        return next_local.astimezone(timezone.utc).isoformat()


__all__ = [
    "SCHEDULER_TASK_END",
    "SCHEDULER_TASK_START",
    "ScheduledTask",
    "TaskScheduler",
]
