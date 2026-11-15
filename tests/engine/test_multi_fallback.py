"""MultiEngine cloud fallback: local model down -> cloud model, with notice."""

from __future__ import annotations

import asyncio

import pytest

from openjarvis.engine._base import (
    EngineConnectionError,
    EngineContextLengthError,
    InferenceEngine,
)
from openjarvis.engine._stubs import StreamChunk
from openjarvis.engine.multi import MultiEngine, is_billing_error

NOTICE = "⚠️ PC apagado, respondiendo con {model}"
BILLING = "Me he quedado sin saldo de Claude ({model})."


class FakeLocal(InferenceEngine):
    engine_id = "ollama"
    is_cloud = False

    def __init__(self, healthy=True, fail_on_call=None, models=("qwen3.5:9b",)):
        self.healthy = healthy
        self.fail_on_call = fail_on_call
        self.models = list(models)
        self.health_calls = 0
        self.calls = []

    def health(self):
        self.health_calls += 1
        return self.healthy

    def list_models(self):
        return self.models if self.healthy else []

    def generate(self, messages, *, model, **kw):
        self.calls.append(model)
        if self.fail_on_call:
            raise self.fail_on_call
        return {"content": f"local:{model}"}

    async def stream(self, messages, *, model, **kw):
        if self.fail_on_call:
            raise self.fail_on_call
        for t in ("lo", "cal"):
            yield t

    async def stream_full(self, messages, *, model, **kw):
        if self.fail_on_call:
            raise self.fail_on_call
        yield StreamChunk(content="local")

    def close(self):
        pass


class FakeCloud(InferenceEngine):
    engine_id = "cloud"
    is_cloud = True

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def health(self):
        return True

    def list_models(self):
        return ["claude-haiku-5-5"]

    def generate(self, messages, *, model, **kw):
        self.calls.append(model)
        if self.error:
            raise self.error
        return {"content": f"cloud:{model}", "usage": {}}

    async def stream(self, messages, *, model, **kw):
        if self.error:
            raise self.error
        for t in ("clo", "ud"):
            yield t

    async def stream_full(self, messages, *, model, **kw):
        if self.error:
            raise self.error
        yield StreamChunk(content="cloud")

    def close(self):
        pass


def _multi(local, cloud, fallback=True, **kw):
    opts = (
        dict(
            cloud_fallback_model="claude-haiku-5-5",
            fallback_notice=NOTICE,
            billing_notice=BILLING,
            local_default_model="qwen3.5:9b",
        )
        if fallback
        else {}
    )
    opts.update(kw)
    return MultiEngine([("ollama", local), ("cloud", cloud)], **opts)


def _collect(agen):
    async def run():
        return [x async for x in agen]

    return asyncio.run(run())


# -- generate -------------------------------------------------------------------


def test_healthy_local_is_used_without_notice():
    local, cloud = FakeLocal(), FakeCloud()
    out = _multi(local, cloud).generate([], model="qwen3.5:9b")
    assert out == {"content": "local:qwen3.5:9b"}
    assert cloud.calls == []


def test_local_down_goes_to_cloud_with_notice():
    local, cloud = FakeLocal(), FakeCloud()
    multi = _multi(local, cloud)
    local.healthy = False  # PC switched off after startup
    out = multi.generate([], model="qwen3.5:9b")
    assert cloud.calls == ["claude-haiku-5-5"]
    assert out["content"] == (
        "⚠️ PC apagado, respondiendo con claude-haiku-5-5\n\ncloud:claude-haiku-5-5"
    )
    assert out["fallback_model"] == "claude-haiku-5-5"


def test_local_down_since_startup_unknown_model():
    local, cloud = FakeLocal(healthy=False), FakeCloud()
    out = _multi(local, cloud).generate([], model="qwen3.5:9b")
    assert out["content"].endswith("cloud:claude-haiku-5-5")


def test_connection_error_mid_call_falls_back():
    local = FakeLocal(fail_on_call=EngineConnectionError("refused"))
    cloud = FakeCloud()
    out = _multi(local, cloud).generate([], model="qwen3.5:9b")
    assert out["content"].endswith("cloud:claude-haiku-5-5")


def test_context_length_error_is_not_a_reason_to_fall_back():
    local = FakeLocal(fail_on_call=EngineContextLengthError("too long"))
    with pytest.raises(EngineContextLengthError):
        _multi(local, FakeCloud()).generate([], model="qwen3.5:9b")


def test_without_fallback_behaviour_is_unchanged():
    local = FakeLocal(healthy=False)
    multi = _multi(local, FakeCloud(), fallback=False)
    with pytest.raises(ValueError):
        multi.generate([], model="qwen3.5:9b")
    multi = _multi(
        FakeLocal(fail_on_call=EngineConnectionError("x")), FakeCloud(), fallback=False
    )
    with pytest.raises(EngineConnectionError):
        multi.generate([], model="qwen3.5:9b")


def test_cloud_models_are_never_rerouted():
    local, cloud = FakeLocal(), FakeCloud()
    out = _multi(local, cloud).generate([], model="claude-haiku-5-5")
    assert out == {"content": "cloud:claude-haiku-5-5", "usage": {}}


def test_down_state_is_cached_then_rechecked(monkeypatch):
    import openjarvis.engine.multi as multi_mod

    clock = [1000.0]
    monkeypatch.setattr(multi_mod.time, "monotonic", lambda: clock[0])
    local, cloud = FakeLocal(), FakeCloud()
    multi = _multi(local, cloud, recheck_seconds=30)
    local.healthy = False
    multi.generate([], model="qwen3.5:9b")
    probes = local.health_calls
    multi.generate([], model="qwen3.5:9b")
    assert local.health_calls == probes  # cached: no new probe
    clock[0] += 31
    local.healthy = True
    out = multi.generate([], model="qwen3.5:9b")
    assert out == {"content": "local:qwen3.5:9b"}  # PC is back


def test_billing_error_becomes_clear_message():
    err = Exception(
        "Error code: 400 - Your credit balance is too low to access the API"
    )
    local, cloud = FakeLocal(healthy=False), FakeCloud(error=err)
    out = _multi(local, cloud).generate([], model="qwen3.5:9b")
    assert out["content"] == "Me he quedado sin saldo de Claude (claude-haiku-5-5)."


def test_other_cloud_errors_propagate():
    local, cloud = FakeLocal(healthy=False), FakeCloud(error=RuntimeError("bad key"))
    with pytest.raises(RuntimeError):
        _multi(local, cloud).generate([], model="qwen3.5:9b")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (Exception("Your credit balance is too low"), True),
        (Exception("insufficient_quota"), True),
        (type("E", (Exception,), {"status_code": 402})("x"), True),
        (Exception("invalid x-api-key"), False),
    ],
)
def test_is_billing_error(exc, expected):
    assert is_billing_error(exc) is expected


def test_local_default_model_stays_listed_while_down():
    local, cloud = FakeLocal(healthy=False), FakeCloud()
    assert _multi(local, cloud).list_models()[0] == "qwen3.5:9b"
    assert "qwen3.5:9b" not in _multi(local, cloud, fallback=False).list_models()


# -- streaming ------------------------------------------------------------------


def test_stream_falls_back_with_notice_first():
    local, cloud = FakeLocal(healthy=False), FakeCloud()
    tokens = _collect(_multi(local, cloud).stream([], model="qwen3.5:9b"))
    assert tokens == [
        "⚠️ PC apagado, respondiendo con claude-haiku-5-5\n\n",
        "clo",
        "ud",
    ]


def test_stream_uses_local_when_healthy():
    tokens = _collect(_multi(FakeLocal(), FakeCloud()).stream([], model="qwen3.5:9b"))
    assert tokens == ["lo", "cal"]


def test_stream_connection_error_before_output_falls_back():
    local = FakeLocal(fail_on_call=EngineConnectionError("refused"))
    tokens = _collect(_multi(local, FakeCloud()).stream([], model="qwen3.5:9b"))
    assert tokens[-2:] == ["clo", "ud"]


def test_stream_billing_error():
    err = Exception("credit balance is too low")
    tokens = _collect(
        _multi(FakeLocal(healthy=False), FakeCloud(error=err)).stream(
            [], model="qwen3.5:9b"
        )
    )
    assert tokens[-1] == "Me he quedado sin saldo de Claude (claude-haiku-5-5)."


def test_stream_full_falls_back():
    chunks = _collect(
        _multi(FakeLocal(healthy=False), FakeCloud()).stream_full(
            [], model="qwen3.5:9b"
        )
    )
    assert [c.content for c in chunks] == [
        "⚠️ PC apagado, respondiendo con claude-haiku-5-5\n\n",
        "cloud",
    ]


# -- config -------------------------------------------------------------------


def test_config_defaults_off_and_toml(tmp_path):
    from openjarvis.core.config import JarvisConfig, load_config

    assert JarvisConfig().engine.cloud_fallback.enabled is False
    path = tmp_path / "config.toml"
    path.write_text(
        '[engine.cloud_fallback]\nenabled = true\nmodel = "claude-haiku-5-5"\n'
        'notice = "aviso {model}"\n',
        encoding="utf-8",
    )
    cf = load_config(path).engine.cloud_fallback
    assert (cf.enabled, cf.model, cf.notice) == (
        True,
        "claude-haiku-5-5",
        "aviso {model}",
    )


# -- cloud model as the default (no fallback) ---------------------------------


def test_cloud_default_billing_error_is_a_clear_reply():
    err = Exception("Your credit balance is too low")
    cloud = FakeCloud(error=err)
    multi = MultiEngine([("cloud", cloud)], billing_notice=BILLING)
    out = multi.generate([], model="claude-haiku-5-5")
    assert out["content"] == "Me he quedado sin saldo de Claude (claude-haiku-5-5)."


def test_cloud_default_billing_error_in_streams():
    err = Exception("credit balance is too low")
    multi = MultiEngine([("cloud", FakeCloud(error=err))], billing_notice=BILLING)
    assert _collect(multi.stream([], model="claude-haiku-5-5")) == [
        "Me he quedado sin saldo de Claude (claude-haiku-5-5)."
    ]
    chunks = _collect(multi.stream_full([], model="claude-haiku-5-5"))
    assert [c.content for c in chunks] == [
        "Me he quedado sin saldo de Claude (claude-haiku-5-5)."
    ]


def test_cloud_default_other_errors_and_no_notice_raise():
    multi = MultiEngine(
        [("cloud", FakeCloud(error=RuntimeError("bad key")))], billing_notice=BILLING
    )
    with pytest.raises(RuntimeError):
        multi.generate([], model="claude-haiku-5-5")
    multi = MultiEngine(
        [("cloud", FakeCloud(error=Exception("credit balance is too low")))]
    )
    with pytest.raises(Exception, match="credit balance"):
        multi.generate([], model="claude-haiku-5-5")


def test_engine_billing_notice_config(tmp_path):
    from openjarvis.core.config import JarvisConfig, load_config

    assert JarvisConfig().engine.billing_notice == ""
    path = tmp_path / "config.toml"
    path.write_text(
        '[engine]\ndefault = "cloud"\nbilling_notice = "sin saldo"\n',
        encoding="utf-8",
    )
    assert load_config(path).engine.billing_notice == "sin saldo"


def test_current_anthropic_models_are_listed():
    from openjarvis.engine.cloud import _ANTHROPIC_MODELS

    assert {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"} <= set(
        _ANTHROPIC_MODELS
    )
