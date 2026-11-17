"""Fase 2: transcripción con faster-whisper + post-procesado premium."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Segment, Transcript

INTERVIEW_NOISE_RE = re.compile(
    r"(^subt[íi]tulos\b|^subtitles\b|amara\.org|susbcr[íi]bete)", re.IGNORECASE
)
FILLERS_RE = re.compile(
    r"\b(m+|eh+|em+|uh+|um+|este|esto|esta|bueno\s+pues)\b[,;:]\s*", re.IGNORECASE
)
SENTENCE_END = re.compile(r"[.!?…]\s*$")


def _shim_pyav() -> None:
    """PyAV ≥ 14 eliminó `metadata_errors` que faster-whisper aún pasa.

    Envuelve av.open para descartar el kwarg si no está soportado, sin
    degradar el resto del comportamiento.
    """
    import functools
    import av
    from av.container import core

    core_open = core.open

    @functools.wraps(core_open)
    def wrapper(*args, **kwargs):
        try:
            return core_open(*args, **kwargs)
        except TypeError as exc:
            if "metadata_errors" not in str(exc):
                raise
            kwargs.pop("metadata_errors", None)
            return core_open(*args, **kwargs)

    wrapper._sf_pyav_shim = True
    core.open = wrapper
    if getattr(av, "open", None) is not wrapper:
        av.open = wrapper


def transcribe(audio: Path, *, model_size: str = "small", device: str = "auto",
               compute_type: str | None = None, language: str | None = None,
               on_progress=None) -> Transcript:
    """Wrapper de faster-whisper. Import perezoso: error accionable si falta."""
    _shim_pyav()
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "faster-whisper no está instalado. Instálalo con `pip install faster-whisper` "
            "(no requiere torch)."
        ) from exc
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    it, info = model.transcribe(str(audio), beam_size=5, vad_filter=True,
                                language=language if language != "auto" else None)
    segments: list[Segment] = []
    for seg in it:
        segments.append(Segment(seg.start, seg.end, seg.text or "",
                                seg.no_speech_prob, seg.avg_logprob))
        if on_progress:
            on_progress(seg.end, info.duration)
    return Transcript(segments, language=info.language or "unknown",
                      model=model_size, duration=info.duration or 0.0)


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# ----------------------------- post-procesado ----------------------------- #

def filter_hallucinations(segments: list[Segment], *, no_speech: float = 0.6,
                          max_logprob: float = -1.0) -> list[Segment]:
    out = []
    for s in segments:
        if s.no_speech_prob > no_speech or s.avg_logprob < max_logprob:
            continue
        if INTERVIEW_NOISE_RE.search(s.text.strip()):
            continue
        out.append(s)
    return out


def dedupe_consecutive(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        t = s.text.strip()
        if out and t and t == out[-1].text.strip():
            out[-1].end = s.end
            continue
        out.append(s)
    return out


def to_paragraphs(segments: list[Segment], *, pause_s: float = 1.5,
                  max_sentences: int = 3) -> str:
    """Párrafos: corte por pausa larga o bloque de frases ya denso."""
    paragraphs: list[str] = []
    current: list[str] = []
    prev = None
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        if prev is not None and current and (s.start - prev.end) > pause_s:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        if len(current) >= max_sentences and SENTENCE_END.search(text):
            paragraphs.append(" ".join(current))
            current = []
        prev = s
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(p for p in paragraphs if p)


def clean_fillers(text: str) -> str:
    """Heurística sin LLM para muletillas comunes de español/inglés."""
    cleaned = FILLERS_RE.sub("", text)
    return re.sub(r" +", " ", cleaned).strip()


def _srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[Segment]) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{_srt_ts(s.start)} --> {_srt_ts(s.end)}", s.text.strip(), ""]
    return "\n".join(lines)


def save_outputs(t: Transcript, outdir: Path, *, clean: bool = False) -> dict[str, Path]:
    """Escribe txt/srt/json y devuelve las rutas."""
    outdir.mkdir(parents=True, exist_ok=True)
    segs = dedupe_consecutive(filter_hallucinations(t.segments))
    paragraphs = to_paragraphs(segs)
    if clean:
        paragraphs = clean_fillers(paragraphs)
    paths = {
        "txt": outdir / "transcripcion.txt",
        "srt": outdir / "transcripcion.srt",
        "json": outdir / "transcripcion.json",
    }
    paths["txt"].write_text(paragraphs + "\n", encoding="utf-8")
    paths["srt"].write_text(to_srt(segs), encoding="utf-8")
    payload = {"language": t.language, "model": t.model, "duration": t.duration,
               "segments": [vars(s) for s in segs]}
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    return paths
