"""Voice-note helpers for channel adapters.

Channels stay unaware of which speech backends exist: they accept two plain
callables (``transcriber`` and ``synthesizer``) and the server builds them from
``[speech]`` config via :func:`configure_channel_voice`.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: ``(audio_bytes, format) -> transcript``
Transcriber = Callable[[bytes, str], str]
#: ``text -> OGG/Opus bytes`` ready to upload as a voice note
Synthesizer = Callable[[str], bytes]

VOICE_REPLY_MODES = ("off", "when_voice", "always")

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_MD_MARKERS = re.compile(r"(^|\s)[#>]+\s*|[*_~]+", re.MULTILINE)
_LIST_BULLET = re.compile(r"^\s*(?:[-+]|\d+[.)])\s+", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[.!?…](?=\s|$)")


def text_for_speech(text: str, max_chars: int = 1000) -> str:
    """Turn a Markdown chat reply into something worth reading aloud.

    Code blocks and URLs are dropped (spoken they are noise), Markdown markers
    are stripped, and the result is cut at the last sentence end that fits in
    *max_chars* so a long answer does not become a minutes-long voice note.
    The full text is always sent as a normal message as well.
    """
    spoken = _CODE_BLOCK.sub(" ", text)
    spoken = _MD_LINK.sub(r"\1", spoken)
    spoken = _INLINE_CODE.sub(r"\1", spoken)
    spoken = _URL.sub(" ", spoken)
    spoken = _LIST_BULLET.sub("", spoken)
    spoken = _MD_MARKERS.sub(r"\1", spoken)
    spoken = _WHITESPACE.sub(" ", spoken).strip()
    if max_chars <= 0 or len(spoken) <= max_chars:
        return spoken
    head = spoken[:max_chars]
    ends = [m.end() for m in _SENTENCE_END.finditer(head)]
    if ends:
        return head[: ends[-1]].strip()
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else head).strip()


def wav_to_ogg_opus(wav: bytes) -> bytes:
    """Re-encode WAV bytes as mono 48 kHz OGG/Opus.

    Telegram only renders an upload as a voice note (waveform, inline player)
    when it is OGG/Opus. PyAV ships its own FFmpeg build, so this needs no
    ``ffmpeg`` binary on PATH.
    """
    import av

    out_buf = io.BytesIO()
    with (
        av.open(io.BytesIO(wav), format="wav") as src,
        av.open(out_buf, mode="w", format="ogg") as dst,
    ):
        stream = dst.add_stream("libopus", rate=48000, layout="mono")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        for frame in src.decode(audio=0):
            for resampled in resampler.resample(frame):
                for packet in stream.encode(resampled):
                    dst.mux(packet)
        for resampled in resampler.resample(None):
            for packet in stream.encode(resampled):
                dst.mux(packet)
        for packet in stream.encode(None):
            dst.mux(packet)
    return out_buf.getvalue()


def make_transcriber(stt_backend: Any, language: str = "") -> Transcriber:
    """Wrap a :class:`~openjarvis.speech._stubs.SpeechBackend` as a callable."""

    def _transcribe(audio: bytes, fmt: str) -> str:
        result = stt_backend.transcribe(audio, format=fmt, language=language or None)
        return (result.text or "").strip()

    return _transcribe


def make_synthesizer(
    tts_backend: Any,
    *,
    voice_id: str = "",
    speed: float = 1.0,
) -> Synthesizer:
    """Wrap a :class:`~openjarvis.speech.tts.TTSBackend` as a callable."""

    def _synthesize(text: str) -> bytes:
        result = tts_backend.synthesize(
            text, voice_id=voice_id, speed=speed, output_format="wav"
        )
        if not result.audio:
            raise RuntimeError(f"{tts_backend.backend_id} returned no audio")
        return wav_to_ogg_opus(result.audio)

    return _synthesize


def configure_channel_voice(
    channel: Any,
    config: Any,
    stt_backend: Optional[Any] = None,
    *,
    tts_factory: Optional[Callable[[str], Optional[Any]]] = None,
) -> str:
    """Attach speech callables to *channel* when its config asks for voice.

    Missing backends only disable the affected direction (with a warning);
    text messaging keeps working either way. Returns a short description of
    what was enabled (e.g. ``"in: faster-whisper, out: kokoro"``), or ``""``.
    """
    set_voice = getattr(channel, "set_voice", None)
    if not callable(set_voice):
        return ""
    wants_input = bool(getattr(channel, "voice_input_enabled", False))
    wants_replies = getattr(channel, "voice_replies", "off") != "off"
    if not (wants_input or wants_replies):
        return ""

    speech = config.speech
    transcriber = None
    if wants_input:
        if stt_backend is None:
            logger.warning(
                "%s voice_input is on but no speech-to-text backend is available",
                channel.channel_id,
            )
        else:
            transcriber = make_transcriber(stt_backend, speech.language)

    synthesizer = None
    if wants_replies:
        if tts_factory is None:
            from openjarvis.speech._tts_discovery import get_tts_backend

            tts_factory = get_tts_backend
        tts_backend = tts_factory(speech.tts_backend)
        if tts_backend is None:
            logger.warning(
                "%s voice_replies is on but no text-to-speech backend is available",
                channel.channel_id,
            )
        else:
            # A voice ID only means something to the backend it was written
            # for; if discovery fell back to another one, use its default.
            same_backend = tts_backend.backend_id == speech.tts_backend
            synthesizer = make_synthesizer(
                tts_backend,
                voice_id=speech.voice_id if same_backend else "",
                speed=speech.voice_speed,
            )

    set_voice(transcriber=transcriber, synthesizer=synthesizer)
    parts = []
    if transcriber is not None:
        parts.append(f"in: {stt_backend.backend_id}")
    if synthesizer is not None:
        parts.append(f"out: {tts_backend.backend_id}")
    return ", ".join(parts)


__all__ = [
    "Synthesizer",
    "Transcriber",
    "VOICE_REPLY_MODES",
    "configure_channel_voice",
    "make_synthesizer",
    "make_transcriber",
    "text_for_speech",
    "wav_to_ogg_opus",
]
