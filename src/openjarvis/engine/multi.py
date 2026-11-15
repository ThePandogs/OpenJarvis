"""Multi-engine wrapper — routes requests to the right backend by model name."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List, Optional

from openjarvis.core.types import Message
from openjarvis.engine._base import (
    EngineConnectionError,
    EngineContextLengthError,
    InferenceEngine,
)
from openjarvis.engine._stubs import StreamChunk

logger = logging.getLogger(__name__)

# Provider error texts that mean "no credit", as opposed to a bad request.
_BILLING_MARKERS = (
    "credit balance",
    "billing",
    "insufficient_quota",
    "insufficient funds",
    "payment required",
    "quota exceeded",
)


def is_billing_error(exc: BaseException) -> bool:
    """Whether *exc* is a cloud provider refusing for lack of credit."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status == 402:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _BILLING_MARKERS)


class MultiEngine(InferenceEngine):
    """Wraps multiple engines and routes by model name.

    Models from each engine are discovered via ``list_models()``.
    When ``generate()`` or ``stream()`` is called, the model name
    is looked up to find which engine owns it.

    Optional cloud fallback (``cloud_fallback_model``): a request for a local
    model whose engine is unreachable — health check failing, or a connection
    error before any output — is answered by that cloud model instead, with
    ``fallback_notice`` prepended so the user knows. Without it, MultiEngine
    keeps refusing to send local models to the cloud (see ``_engine_for``).
    """

    engine_id = "multi"

    _CLOUD_PREFIXES = ("gpt-", "o1-", "o3-", "o4-", "claude-", "gemini-", "openrouter/")

    def __init__(
        self,
        engines: list[tuple[str, InferenceEngine]],
        *,
        cloud_fallback_model: str = "",
        fallback_notice: str = "",
        billing_notice: str = "",
        local_default_model: str = "",
        recheck_seconds: float = 30.0,
    ) -> None:
        self._engines = engines
        self._model_map: Dict[str, InferenceEngine] = {}
        self._model_key_map: Dict[str, str] = {}
        self._fallback_model = cloud_fallback_model
        self._fallback_notice = fallback_notice
        self._billing_notice = billing_notice
        self._local_default_model = local_default_model
        self._recheck_seconds = recheck_seconds
        self._local_down_until = 0.0
        self._billing_warned = False
        self._lock = threading.Lock()
        self._refresh_map()

    def _refresh_map(self) -> None:
        self._model_map.clear()
        self._model_key_map.clear()
        for key, engine in self._engines:
            try:
                for model_id in engine.list_models():
                    self._model_map[model_id] = engine
                    self._model_key_map[model_id] = key
            except Exception as exc:
                logger.debug("Failed to list models for %s: %s", key, exc)

    def _engine_for(self, model: str) -> InferenceEngine:
        """Find the engine that owns a model, refreshing the map once if needed."""
        engine = self._model_map.get(model)
        if engine is not None:
            return engine
        # Refresh and retry (a new model may have been pulled)
        self._refresh_map()
        engine = self._model_map.get(model)
        if engine is not None:
            return engine
        # If model looks like a cloud model, route to the cloud engine
        # rather than falling back to the local engine (which would 404).
        if any(model.startswith(p) for p in self._CLOUD_PREFIXES):
            for key, eng in self._engines:
                if key == "cloud":
                    logger.info("Routing cloud model %r to cloud engine", model)
                    return eng
        # Non-cloud models: do NOT silently fall back to cloud. A transient
        # vLLM outage during a long agentic run would otherwise route every
        # call to cloud, producing confusing "invalid model ID" errors
        # across all tasks.
        raise ValueError(
            f"Model {model!r} not found in any engine "
            f"(known: {', '.join(sorted(self._model_map.keys())) or '<none>'}). "
            f"Check that the expected backend (e.g. vLLM server) is reachable."
        )

    # -- cloud fallback ----------------------------------------------------------

    def _cloud_engine(self) -> Optional[InferenceEngine]:
        for key, eng in self._engines:
            if key == "cloud":
                return eng
        return None

    def _fallback_ready(self, model: str) -> bool:
        return bool(
            self._fallback_model
            and not any(model.startswith(p) for p in self._CLOUD_PREFIXES)
            and self._cloud_engine() is not None
        )

    def _mark_local_down(self) -> None:
        with self._lock:
            self._local_down_until = time.monotonic() + self._recheck_seconds

    def _local_unavailable(self, model: str) -> bool:
        """True when *model*'s local engine cannot serve right now.

        Health checks are short (Ollama: 2 s) but would still add that delay
        to every message while the PC is off, so a failure is remembered for
        ``recheck_seconds``.
        """
        with self._lock:
            if time.monotonic() < self._local_down_until:
                return True
        owner = self._model_map.get(model)
        if owner is not None and not getattr(owner, "is_cloud", False):
            healthy = _safe_health(owner)
        else:
            # Unknown model: if no local engine answers, it cannot be served
            # locally; if one does, let normal routing decide (and maybe 404).
            healthy = any(
                _safe_health(eng)
                for key, eng in self._engines
                if key != "cloud" and not getattr(eng, "is_cloud", False)
            )
        if not healthy:
            self._mark_local_down()
        return not healthy

    def _notice(self) -> str:
        return (self._fallback_notice or "").format(model=self._fallback_model)

    def _billing_text(self, exc: BaseException, model: str = "") -> str:
        model = model or self._fallback_model
        if not self._billing_warned:
            self._billing_warned = True
            logger.warning(
                "Cloud model %s refused for lack of credit: %s",
                model,
                type(exc).__name__,
            )
        return (self._billing_notice or str(exc)).format(model=model)

    def _billing_reply(self, exc: BaseException, model: str) -> Optional[str]:
        """Reply text for a lack-of-credit refusal, or None to re-raise."""
        if self._billing_notice and is_billing_error(exc):
            return self._billing_text(exc, model)
        return None

    def _cloud_generate(self, messages, kwargs) -> Dict[str, Any]:
        cloud = self._cloud_engine()
        try:
            result = cloud.generate(messages, model=self._fallback_model, **kwargs)
        except Exception as exc:
            if self._billing_notice and is_billing_error(exc):
                return {"content": self._billing_text(exc), "finish_reason": "stop"}
            raise
        result = dict(result)
        notice = self._notice()
        if notice:
            result["content"] = f"{notice}\n\n{result.get('content') or ''}".rstrip()
        result["fallback_model"] = self._fallback_model
        return result

    # -- InferenceEngine ----------------------------------------------------------

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        call = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        if self._fallback_ready(model) and self._local_unavailable(model):
            return self._cloud_generate(messages, call)
        try:
            engine = self._local_engine_for(model)
            return engine.generate(messages, model=model, **call)
        except EngineContextLengthError:
            raise
        except EngineConnectionError:
            if not self._fallback_ready(model):
                raise
            self._mark_local_down()
            logger.warning(
                "Local model %r unreachable; using %s", model, self._fallback_model
            )
            return self._cloud_generate(messages, call)
        except Exception as exc:
            # A cloud model used directly (not as fallback) running out of
            # credit gets the same clear message instead of a stack trace.
            reply = self._billing_reply(exc, model)
            if reply is None:
                raise
            return {"content": reply, "finish_reason": "stop"}

    def _local_engine_for(self, model: str) -> InferenceEngine:
        """``_engine_for``, but a model no engine lists counts as unreachable
        when a fallback exists (its engine dropped it from the map because it
        is down), so the caller's EngineConnectionError path handles it."""
        try:
            return self._engine_for(model)
        except ValueError as exc:
            if self._fallback_ready(model):
                raise EngineConnectionError(str(exc)) from exc
            raise

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        call = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        use_cloud = self._fallback_ready(model) and self._local_unavailable(model)
        if not use_cloud:
            yielded = False
            try:
                async for token in self._local_engine_for(model).stream(
                    messages, model=model, **call
                ):
                    yielded = True
                    yield token
                return
            except EngineContextLengthError:
                raise
            except EngineConnectionError:
                # Once output has started, switching models mid-answer would
                # produce a spliced reply: only fall back before the first token.
                if yielded or not self._fallback_ready(model):
                    raise
                self._mark_local_down()
            except Exception as exc:
                reply = None if yielded else self._billing_reply(exc, model)
                if reply is None:
                    raise
                yield reply
                return
        async for token in self._cloud_stream(messages, call):
            yield token

    async def _cloud_stream(self, messages, call) -> AsyncIterator[str]:
        notice = self._notice()
        if notice:
            yield f"{notice}\n\n"
        try:
            async for token in self._cloud_engine().stream(
                messages, model=self._fallback_model, **call
            ):
                yield token
        except Exception as exc:
            if self._billing_notice and is_billing_error(exc):
                yield self._billing_text(exc)
                return
            raise

    async def stream_full(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator["StreamChunk"]:
        """Delegate stream_full() to the engine that owns the model."""
        use_cloud = self._fallback_ready(model) and self._local_unavailable(model)
        if not use_cloud:
            yielded = False
            try:
                engine = self._local_engine_for(model)
                async for chunk in engine.stream_full(messages, model=model, **kwargs):
                    yielded = True
                    yield chunk
                return
            except EngineContextLengthError:
                raise
            except EngineConnectionError:
                if yielded or not self._fallback_ready(model):
                    raise
                self._mark_local_down()
            except Exception as exc:
                reply = None if yielded else self._billing_reply(exc, model)
                if reply is None:
                    raise
                yield StreamChunk(content=reply, finish_reason="stop")
                return
        notice = self._notice()
        if notice:
            yield StreamChunk(content=f"{notice}\n\n")
        try:
            async for chunk in self._cloud_engine().stream_full(
                messages, model=self._fallback_model, **kwargs
            ):
                yield chunk
        except Exception as exc:
            if self._billing_notice and is_billing_error(exc):
                yield StreamChunk(content=self._billing_text(exc), finish_reason="stop")
                return
            raise

    def list_models(self) -> List[str]:
        self._refresh_map()
        models = list(self._model_map.keys())
        # With a fallback the local default model stays servable (by the
        # cloud) while its engine is down, so keep it selectable in pickers.
        if (
            self._fallback_model
            and self._local_default_model
            and self._local_default_model not in models
        ):
            models.insert(0, self._local_default_model)
        return models

    def engine_key_for(self, model: str) -> str | None:
        """Return the registry key of the engine advertising *model*."""
        key = self._model_key_map.get(model)
        if key is not None:
            return key
        self._refresh_map()
        return self._model_key_map.get(model)

    def health(self) -> bool:
        return any(engine.health() for _key, engine in self._engines)

    def close(self) -> None:
        for _key, engine in self._engines:
            engine.close()


def _safe_health(engine: InferenceEngine) -> bool:
    try:
        return bool(engine.health())
    except Exception:  # noqa: BLE001
        return False


__all__ = ["MultiEngine", "is_billing_error"]
