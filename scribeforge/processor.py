"""Fase 1: FFmpeg — validación, extracción y segmentación de audio."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"

SUPPORTED_EXTS = {".mp4", ".mkv", ".mov", ".mp3", ".wav", ".m4a", ".flac", ".opus", ".ogg", ".webm"}

# Umbral de duración (s) por encima del cual se trocea el audio en bloques.
SEGMENT_THRESHOLD_S = 2 * 3600
BLOCK_S = 30 * 60
BLOCK_OVERLAP_S = 2.0


class ProcessorError(RuntimeError):
    pass


def check_ffmpeg() -> None:
    """Fallar con mensaje accionable si ffmpeg no está instalado."""
    if not shutil.which(FFMPEG):
        raise ProcessorError(
            "ffmpeg no encontrado en PATH. Instálalo (ej. `apt install ffmpeg` "
            "o https://ffmpeg.org/download.html) y reintenta."
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_duration(path: Path) -> float:
    """Duración en segundos leyendo la cabecera que ffmpeg vuelca a stderr."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, check=False,
    )
    for line in proc.stderr.splitlines():
        if "Duration:" in line:
            stamp = line.split("Duration:")[1].split(",")[0].strip()
            parts = [float(p) for p in stamp.split(":")]
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ProcessorError(f"No se pudo leer la duración de {path}. ¿Formato corrupto o no soportado?")


def validate_input(path: Path) -> None:
    if not path.exists():
        raise ProcessorError(f"El archivo de entrada no existe: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ProcessorError(
            f"Extensión no soportada '{path.suffix}'. Soportadas: {', '.join(sorted(SUPPORTED_EXTS))}"
        )


def extract_audio(input_path: Path, workdir: Path) -> Path:
    """Extrae a WAV 16 kHz mono PCM s16, silencioso y con -y para idempotencia."""
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "audio.wav"
    proc = subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(input_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         str(out)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0 or not out.exists():
        raise ProcessorError(f"ffmpeg falló al extraer audio de {input_path}:\n{proc.stderr.strip()}")
    return out


def audio_blocks(audio: Path, duration_s: float) -> list[tuple[Path, float]]:
    """(bloque, offset) — devuelve [audio] completo si no supera el umbral.

    Para audios muy largos trocea en bloques de BLOCK_S con solape de
    BLOCK_OVERLAP_S para no cortar palabras a mitad.
    """
    if duration_s <= SEGMENT_THRESHOLD_S:
        return [(audio, 0.0)]
    blocks: list[tuple[Path, float]] = []
    start = 0.0
    i = 0
    while start < duration_s:
        out = audio.parent / f"audio.part{i:03d}.wav"
        proc = subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(audio), "-ss", str(start), "-t", str(BLOCK_S),
             "-c", "copy", str(out)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise ProcessorError(f"ffmpeg falló troceando {audio} (bloque {i}):\n{proc.stderr.strip()}")
        blocks.append((out, start))
        start += BLOCK_S - BLOCK_OVERLAP_S
        i += 1
    return blocks
