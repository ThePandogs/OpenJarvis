"""Morning digest: language, verbatim status footer, and spoken text for delivery."""

from __future__ import annotations

import os
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

from openjarvis.core.types import ToolResult


def _agent(tmp_path, **kw):
    from openjarvis.agents.morning_digest import MorningDigestAgent

    engine = MagicMock()
    engine.generate.return_value = {
        "content": "Buenos días, Carlos. Hoy en Vigo llueve.",
        "finish_reason": "stop",
        "usage": {},
    }
    agent = MorningDigestAgent(
        engine,
        "test-model",
        tools=[],
        sections=["world"],
        section_sources={"world": ["open_meteo"]},
        digest_store_path=str(tmp_path / "digest.db"),
        **kw,
    )
    return agent, engine


def _run(agent, tmp_path):
    collect = ToolResult(
        tool_name="digest_collect",
        content="=== WORLD ===\n[weather] Vigo hoy: lluvia\n",
        success=True,
    )
    tts = ToolResult(
        tool_name="text_to_speech",
        content="",
        success=True,
        metadata={"audio_path": str(tmp_path / "d.wav")},
    )
    # The agent's optional self-evaluation imports a DigestEvaluator that this
    # tree does not ship, so that step is skipped and generate runs once.
    with patch.object(agent._executor, "execute", side_effect=[collect, tts]):
        return agent.run("digest")


def test_language_instruction_only_when_configured(tmp_path):
    agent, engine = _agent(tmp_path, language="Spanish")
    _run(agent, tmp_path)
    system = engine.generate.call_args.args[0][0].text
    assert "Write the whole briefing in Spanish" in system

    agent, engine = _agent(tmp_path)
    _run(agent, tmp_path)
    assert (
        "Write the whole briefing in" not in engine.generate.call_args.args[0][0].text
    )


def test_status_footer_is_verbatim_and_not_spoken(tmp_path):
    status = tmp_path / "vigilante_estado.txt"
    status.write_text("﻿NAS OK - comprobado 07:55\n", encoding="utf-8")
    agent, _ = _agent(tmp_path, status_files=[str(status)])
    result = _run(agent, tmp_path)
    assert result.content == (
        "Buenos días, Carlos. Hoy en Vigo llueve.\n\nNAS OK - comprobado 07:55"
    )
    assert result.metadata["spoken_text"] == "Buenos días, Carlos. Hoy en Vigo llueve."
    assert result.metadata["status_footer"] == "NAS OK - comprobado 07:55"
    assert result.metadata["audio_path"].endswith("d.wav")


def test_stale_status_is_flagged(tmp_path):
    status = tmp_path / "vigilante_estado.txt"
    status.write_text("NAS OK - comprobado 03:10\n", encoding="utf-8")
    old = time.time() - 3 * 3600
    os.utime(status, (old, old))
    agent, _ = _agent(tmp_path, status_files=[str(status)], status_max_age_minutes=30)
    footer = agent._status_footer()
    stamp = datetime.fromtimestamp(old).strftime("%d/%m %H:%M")
    assert footer == f"⚠ NAS OK - comprobado 03:10 (sin actualizar desde {stamp})"


def test_missing_status_file_is_reported(tmp_path):
    agent, _ = _agent(tmp_path, status_files=[str(tmp_path / "no-existe.txt")])
    assert agent._status_footer() == "⚠ no-existe.txt: no disponible"


def test_no_status_files_no_footer(tmp_path):
    agent, _ = _agent(tmp_path)
    result = _run(agent, tmp_path)
    assert result.content == "Buenos días, Carlos. Hoy en Vigo llueve."
    assert result.metadata["status_footer"] == ""


def test_orchestrator_passes_digest_config(tmp_path):
    from openjarvis.core.config import DigestConfig

    dc = DigestConfig(
        language="Spanish", status_files=["x.txt"], status_max_age_minutes=45
    )
    assert dc.language == "Spanish" and dc.status_files == ["x.txt"]
    # The kwargs the orchestrator builds must be accepted by the agent.
    agent, _ = _agent(
        tmp_path,
        language=dc.language,
        status_files=dc.status_files,
        status_max_age_minutes=dc.status_max_age_minutes,
    )
    assert agent._status_max_age_minutes == 45
