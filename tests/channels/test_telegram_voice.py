"""Tests for Telegram voice notes (inbound transcription, outbound voice replies)."""

from __future__ import annotations

import io
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.channels._voice import (
    configure_channel_voice,
    make_synthesizer,
    make_transcriber,
    text_for_speech,
)
from openjarvis.channels.telegram import TelegramChannel, _audio_format
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_telegram():
    if not ChannelRegistry.contains("telegram"):
        ChannelRegistry.register_value("telegram", TelegramChannel)


def _ok_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "ok"
    return resp


class _Filter:
    """Stand-in for PTB filters: supports ``|`` like the real ones."""

    def __init__(self, name):
        self.name = name

    def __or__(self, other):
        return _Filter(f"{self.name}|{other.name}")


# -- text_for_speech -----------------------------------------------------------


class TestTextForSpeech:
    def test_strips_markdown_code_and_urls(self):
        text = (
            "## Resumen\n"
            "Tienes **3 correos** importantes:\n"
            "- uno de `Ana`\n"
            "- otro con [enlace](https://example.com)\n"
            "```python\nprint('x')\n```\n"
            "Mira https://example.org/x para más."
        )
        spoken = text_for_speech(text)
        assert "**" not in spoken and "`" not in spoken and "#" not in spoken
        assert "print" not in spoken
        assert "http" not in spoken
        assert "Tienes 3 correos importantes" in spoken
        assert "uno de Ana" in spoken
        assert "otro con enlace" in spoken

    def test_cuts_long_text_at_sentence_end(self):
        text = "Primera frase corta. Segunda frase bastante más larga que sigue."
        assert text_for_speech(text, max_chars=30) == "Primera frase corta."

    def test_cuts_at_word_when_no_sentence_end(self):
        spoken = text_for_speech("palabra " * 20, max_chars=25)
        assert len(spoken) <= 25
        assert not spoken.endswith(" ")
        assert spoken.split() == ["palabra"] * len(spoken.split())

    def test_zero_limit_means_no_cut(self):
        text = "a. " * 600
        assert len(text_for_speech(text, max_chars=0)) > 1000

    def test_empty_after_cleaning(self):
        assert text_for_speech("```\nonly code\n```") == ""


# -- wav_to_ogg_opus -----------------------------------------------------------


def test_wav_to_ogg_opus_produces_voice_note():
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from openjarvis.channels._voice import wav_to_ogg_opus

    t = np.linspace(0, 1.0, 24000, endpoint=False)
    buf = io.BytesIO()
    sf.write(
        buf, (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 24000, format="WAV"
    )
    ogg = wav_to_ogg_opus(buf.getvalue())

    assert ogg[:4] == b"OggS"
    with av.open(io.BytesIO(ogg)) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "opus"
        assert stream.rate == 48000
        assert stream.layout.name == "mono"


# -- callables -----------------------------------------------------------------


def test_transcriber_passes_language_and_strips():
    stt = MagicMock()
    stt.transcribe.return_value = SimpleNamespace(text="  hola  ")
    assert make_transcriber(stt, "es")(b"audio", "ogg") == "hola"
    stt.transcribe.assert_called_once_with(b"audio", format="ogg", language="es")


def test_transcriber_autodetects_when_language_empty():
    stt = MagicMock()
    stt.transcribe.return_value = SimpleNamespace(text="hi")
    make_transcriber(stt, "")(b"a", "ogg")
    assert stt.transcribe.call_args.kwargs["language"] is None


def test_synthesizer_requests_wav_and_converts():
    tts = MagicMock()
    tts.backend_id = "kokoro"
    tts.synthesize.return_value = SimpleNamespace(audio=b"RIFF....")
    with patch(
        "openjarvis.channels._voice.wav_to_ogg_opus", return_value=b"OggS"
    ) as conv:
        out = make_synthesizer(tts, voice_id="ef_dora", speed=1.1)("hola")
    assert out == b"OggS"
    tts.synthesize.assert_called_once_with(
        "hola", voice_id="ef_dora", speed=1.1, output_format="wav"
    )
    conv.assert_called_once_with(b"RIFF....")


def test_synthesizer_rejects_empty_audio():
    tts = MagicMock()
    tts.backend_id = "kokoro"
    tts.synthesize.return_value = SimpleNamespace(audio=b"")
    with pytest.raises(RuntimeError):
        make_synthesizer(tts)("hola")


# -- configure_channel_voice ---------------------------------------------------


def _speech_config(**overrides):
    values = dict(
        language="es", tts_backend="kokoro", voice_id="ef_dora", voice_speed=1.0
    )
    values.update(overrides)
    return SimpleNamespace(speech=SimpleNamespace(**values))


class TestConfigureChannelVoice:
    def test_noop_when_voice_disabled(self):
        ch = TelegramChannel(bot_token="t")
        factory = MagicMock()
        assert (
            configure_channel_voice(
                ch, _speech_config(), MagicMock(), tts_factory=factory
            )
            == ""
        )
        factory.assert_not_called()
        assert ch._transcriber is None and ch._synthesizer is None

    def test_noop_for_channels_without_voice_support(self):
        assert configure_channel_voice(object(), _speech_config(), None) == ""

    def test_enables_both_directions(self):
        ch = TelegramChannel(
            bot_token="t", voice_input=True, voice_replies="when_voice"
        )
        stt = MagicMock(backend_id="faster-whisper")
        tts = MagicMock(backend_id="kokoro")
        desc = configure_channel_voice(
            ch, _speech_config(), stt, tts_factory=lambda _pref: tts
        )
        assert desc == "in: faster-whisper, out: kokoro"
        assert ch._transcriber is not None and ch._synthesizer is not None

    def test_foreign_backend_gets_its_default_voice(self):
        ch = TelegramChannel(bot_token="t", voice_replies="always")
        tts = MagicMock(backend_id="openai_tts")
        tts.synthesize.return_value = SimpleNamespace(audio=b"RIFF")
        configure_channel_voice(
            ch, _speech_config(), None, tts_factory=lambda _pref: tts
        )
        with patch("openjarvis.channels._voice.wav_to_ogg_opus", return_value=b"o"):
            ch._synthesizer("hola")
        assert tts.synthesize.call_args.kwargs["voice_id"] == ""

    def test_missing_backends_warn_and_disable(self, caplog):
        ch = TelegramChannel(bot_token="t", voice_input=True, voice_replies="always")
        with caplog.at_level(logging.WARNING):
            desc = configure_channel_voice(
                ch, _speech_config(), None, tts_factory=lambda _pref: None
            )
        assert desc == ""
        assert ch._transcriber is None and ch._synthesizer is None
        assert "speech-to-text" in caplog.text
        assert "text-to-speech" in caplog.text


# -- config plumbing -----------------------------------------------------------


def test_toml_voice_settings_reach_the_adapter(tmp_path):
    from openjarvis.core.config import load_config
    from openjarvis.core.events import EventBus
    from openjarvis.system.builder import SystemBuilder

    path = tmp_path / "config.toml"
    path.write_text(
        '[channel]\nenabled = true\ndefault_channel = "telegram"\n'
        '[channel.telegram]\nbot_token = "t"\nvoice_input = true\n'
        'voice_replies = "when_voice"\nvoice_max_chars = 300\n'
        'voice_unrecognized_reply = "No te he entendido"\n'
    )
    cfg = load_config(path)
    ch = SystemBuilder(config=cfg)._resolve_channel(cfg, EventBus())
    assert isinstance(ch, TelegramChannel)
    assert ch.voice_input_enabled is True
    assert ch.voice_replies == "when_voice"
    assert ch._voice_max_chars == 300
    assert ch._voice_unrecognized_reply == "No te he entendido"


def test_voice_defaults_are_off():
    ch = TelegramChannel(bot_token="t")
    assert ch.voice_input_enabled is False
    assert ch.voice_replies == "off"


def test_invalid_voice_mode_rejected():
    with pytest.raises(ValueError):
        TelegramChannel(bot_token="t", voice_replies="sometimes")


def test_audio_format_guess():
    assert _audio_format(SimpleNamespace(mime_type="audio/ogg")) == "ogg"
    assert _audio_format(SimpleNamespace(mime_type="audio/mpeg")) == "mp3"
    assert _audio_format(SimpleNamespace(mime_type="", file_name="nota.M4A")) == "m4a"
    assert _audio_format(SimpleNamespace()) == "ogg"


# -- outbound voice replies ----------------------------------------------------


def _voice_channel(mode, synth=None):
    ch = TelegramChannel(bot_token="t", voice_replies=mode, parse_mode="")
    ch.set_voice(synthesizer=synth or MagicMock(return_value=b"OggS-bytes"))
    return ch


def _endpoints(mock_post):
    return [call.args[0].rsplit("/", 1)[-1] for call in mock_post.call_args_list]


class TestVoiceReplies:
    def test_when_voice_answers_voice_note_with_text_then_voice(self):
        ch = _voice_channel("when_voice")
        ch._remember_voice_message("42")
        with patch("httpx.post", return_value=_ok_response()) as post:
            assert ch.send("555", "**Hola**, qué tal.", conversation_id="42")
        assert _endpoints(post) == ["sendMessage", "sendVoice"]
        voice_call = post.call_args_list[1]
        assert voice_call.kwargs["data"] == {
            "chat_id": "555",
            "reply_to_message_id": "42",
        }
        name, payload, mime = voice_call.kwargs["files"]["voice"]
        assert payload == b"OggS-bytes" and mime == "audio/ogg"
        ch._synthesizer.assert_called_once_with("Hola, qué tal.")

    def test_when_voice_answers_once_per_voice_note(self):
        ch = _voice_channel("when_voice")
        ch._remember_voice_message("42")
        with patch("httpx.post", return_value=_ok_response()) as post:
            ch.send("555", "uno", conversation_id="42")
            ch.send("555", "dos", conversation_id="42")
        assert _endpoints(post) == ["sendMessage", "sendVoice", "sendMessage"]

    def test_when_voice_keeps_text_messages_text_only(self):
        ch = _voice_channel("when_voice")
        with patch("httpx.post", return_value=_ok_response()) as post:
            ch.send("555", "hola", conversation_id="7")
        assert _endpoints(post) == ["sendMessage"]
        ch._synthesizer.assert_not_called()

    def test_always_mode(self):
        ch = _voice_channel("always")
        with patch("httpx.post", return_value=_ok_response()) as post:
            ch.send("555", "hola")
        assert _endpoints(post) == ["sendMessage", "sendVoice"]
        assert "reply_to_message_id" not in post.call_args_list[1].kwargs["data"]

    def test_off_mode_never_synthesizes(self):
        ch = _voice_channel("off")
        ch._remember_voice_message("42")
        with patch("httpx.post", return_value=_ok_response()) as post:
            ch.send("555", "hola", conversation_id="42")
        assert _endpoints(post) == ["sendMessage"]

    def test_no_synthesizer_means_text_only(self):
        ch = TelegramChannel(bot_token="t", voice_replies="always", parse_mode="")
        with patch("httpx.post", return_value=_ok_response()) as post:
            assert ch.send("555", "hola")
        assert _endpoints(post) == ["sendMessage"]

    def test_synthesis_failure_still_reports_delivered_text(self, caplog):
        ch = _voice_channel("always", synth=MagicMock(side_effect=RuntimeError("x")))
        with patch("httpx.post", return_value=_ok_response()) as post:
            with caplog.at_level(logging.WARNING):
                assert ch.send("555", "hola") is True
        assert _endpoints(post) == ["sendMessage"]
        assert "voice reply failed" in caplog.text

    def test_send_voice_http_error_is_not_fatal(self):
        ch = _voice_channel("always")
        bad = MagicMock(status_code=400, text="Bad Request")
        with patch("httpx.post", side_effect=[_ok_response(), bad]):
            assert ch.send("555", "hola") is True

    def test_text_failure_skips_voice(self):
        ch = _voice_channel("always")
        bad = MagicMock(status_code=500, text="boom")
        with patch("httpx.post", return_value=bad) as post:
            assert ch.send("555", "hola") is False
        assert _endpoints(post) == ["sendMessage"]
        ch._synthesizer.assert_not_called()

    def test_reply_with_only_code_sends_no_voice(self):
        ch = _voice_channel("always")
        with patch("httpx.post", return_value=_ok_response()) as post:
            ch.send("555", "```\nls -la\n```")
        assert _endpoints(post) == ["sendMessage"]

    def test_voice_id_memory_is_bounded(self):
        ch = _voice_channel("when_voice")
        for i in range(ch._VOICE_IDS_KEPT + 10):
            ch._remember_voice_message(str(i))
        assert len(ch._voice_message_ids) == ch._VOICE_IDS_KEPT
        assert "0" not in ch._voice_message_ids


# -- inbound voice notes -------------------------------------------------------


def _run_poll_loop_with(monkeypatch, channel, updates):
    """Drive _poll_loop with fake PTB objects, delivering *updates* to the
    handler registered for each update's kind ("text" or "voice")."""
    handlers = []
    app = SimpleNamespace(add_handler=handlers.append)
    builder = MagicMock()
    builder.token.return_value = builder
    builder.build.return_value = app
    text_filter, voice_filter, audio_filter = (
        _Filter("TEXT"),
        _Filter("VOICE"),
        _Filter("AUDIO"),
    )
    telegram_ext = SimpleNamespace(
        ApplicationBuilder=lambda: builder,
        MessageHandler=lambda flt, callback: SimpleNamespace(
            filter=flt, callback=callback
        ),
        filters=SimpleNamespace(
            TEXT=text_filter, VOICE=voice_filter, AUDIO=audio_filter
        ),
    )
    monkeypatch.setitem(sys.modules, "telegram.ext", telegram_ext)

    async def deliver(_app, _stop):
        by_kind = {"text" if h.filter is text_filter else "voice": h for h in handlers}
        for kind, update in updates:
            await by_kind[kind].callback(update, None)

    monkeypatch.setattr(channel, "_run_polling_lifecycle", deliver)
    channel._poll_loop()
    return handlers


def _voice_update(chat_id=555, message_id=9, data=b"opus", voice=True):
    tg_file = MagicMock()

    async def download():
        return bytearray(data)

    tg_file.download_as_bytearray = download

    async def get_file():
        return tg_file

    media = SimpleNamespace(
        get_file=MagicMock(side_effect=get_file), mime_type="audio/ogg"
    )
    msg = SimpleNamespace(
        from_user=SimpleNamespace(id=chat_id),
        chat=SimpleNamespace(id=chat_id),
        message_id=message_id,
        voice=media if voice else None,
        audio=None if voice else media,
        text=None,
    )
    return SimpleNamespace(message=msg), media


class TestInboundVoice:
    def test_voice_handler_only_registered_when_enabled(self, monkeypatch):
        handlers = _run_poll_loop_with(monkeypatch, TelegramChannel(bot_token="t"), [])
        assert len(handlers) == 1

    def test_voice_note_is_transcribed_and_dispatched(self, monkeypatch):
        ch = TelegramChannel(bot_token="t", voice_input=True)
        transcriber = MagicMock(return_value="¿qué tiempo hace?")
        ch.set_voice(transcriber=transcriber)
        received = []
        ch.on_message(received.append)
        update, _ = _voice_update(data=b"opus-bytes")

        _run_poll_loop_with(monkeypatch, ch, [("voice", update)])

        transcriber.assert_called_once_with(b"opus-bytes", "ogg")
        assert len(received) == 1
        cm = received[0]
        assert cm.content == "¿qué tiempo hace?"
        assert cm.conversation_id == "555" and cm.message_id == "9"
        assert cm.metadata == {"voice": True}
        assert "9" in ch._voice_message_ids

    def test_unlisted_chat_audio_is_never_downloaded(self, monkeypatch):
        ch = TelegramChannel(bot_token="t", voice_input=True, allowed_chat_ids="111")
        transcriber = MagicMock(return_value="x")
        ch.set_voice(transcriber=transcriber)
        received = []
        ch.on_message(received.append)
        update, media = _voice_update(chat_id=999)

        _run_poll_loop_with(monkeypatch, ch, [("voice", update)])

        media.get_file.assert_not_called()
        transcriber.assert_not_called()
        assert received == []

    def test_empty_transcript_sends_unrecognized_reply(self, monkeypatch):
        ch = TelegramChannel(
            bot_token="t",
            voice_input=True,
            voice_unrecognized_reply="No te he entendido",
        )
        ch.set_voice(transcriber=MagicMock(return_value=""))
        received = []
        ch.on_message(received.append)
        update, _ = _voice_update()

        with patch.object(ch, "send", return_value=True) as send:
            _run_poll_loop_with(monkeypatch, ch, [("voice", update)])

        assert received == []
        send.assert_called_once_with("555", "No te he entendido", conversation_id="9")

    def test_transcription_error_is_contained(self, monkeypatch, caplog):
        ch = TelegramChannel(
            bot_token="t", voice_input=True, voice_unrecognized_reply=""
        )
        ch.set_voice(transcriber=MagicMock(side_effect=OSError("decode")))
        received = []
        ch.on_message(received.append)
        update, _ = _voice_update()

        with caplog.at_level(logging.WARNING):
            _run_poll_loop_with(monkeypatch, ch, [("voice", update)])

        assert received == []
        assert "could not be transcribed: OSError" in caplog.text

    def test_missing_transcriber_warns(self, monkeypatch, caplog):
        ch = TelegramChannel(
            bot_token="t", voice_input=True, voice_unrecognized_reply=""
        )
        update, media = _voice_update()
        with caplog.at_level(logging.WARNING):
            _run_poll_loop_with(monkeypatch, ch, [("voice", update)])
        media.get_file.assert_not_called()
        assert "no speech-to-text backend" in caplog.text

    def test_audio_file_uses_its_mime_type(self, monkeypatch):
        ch = TelegramChannel(bot_token="t", voice_input=True)
        transcriber = MagicMock(return_value="hola")
        ch.set_voice(transcriber=transcriber)
        update, media = _voice_update(voice=False)
        media.mime_type = "audio/mpeg"

        _run_poll_loop_with(monkeypatch, ch, [("voice", update)])

        assert transcriber.call_args.args[1] == "mp3"

    def test_text_messages_still_work_with_voice_enabled(self, monkeypatch):
        ch = TelegramChannel(bot_token="t", voice_input=True)
        received = []
        ch.on_message(received.append)
        update = SimpleNamespace(
            message=SimpleNamespace(
                from_user=SimpleNamespace(id=1),
                text="hola",
                message_id=3,
                chat=SimpleNamespace(id=1),
            )
        )
        _run_poll_loop_with(monkeypatch, ch, [("text", update)])
        assert [m.content for m in received] == ["hola"]
        assert received[0].metadata == {}
