"""Fase 3: chunking con solape, map-reduce y prompts versionados en markdown."""
from __future__ import annotations

from pathlib import Path

from .llm import LLMClient, LLMError
from .models import Transcript

PROMPTS_DIR = Path(__file__).parent / "prompts"
MODES = ("interview", "meeting", "lecture")
CONTEXT_FRAC = 0.7  # % del chunk máximo usado para dejar margen al solape y a la respuesta


def load_prompt(mode: str) -> str:
    path = PROMPTS_DIR / f"{mode}.md"
    if not path.exists():
        raise LLMError(f"Prompt de modo '{mode}' no encontrado en {path}.")
    return path.read_text(encoding="utf-8")


def chunk_by_chars(text: str, max_chars: int, overlap_ratio: float = 0.1) -> list[str]:
    """Chunks ≤ max_chars con solape proporcional. Nunca vacío, nunca > max_chars."""
    if max_chars <= 0:
        raise ValueError("max_chars debe ser positivo")
    if len(text) <= max_chars:
        return [text] if text else []
    overlap = int(max_chars * overlap_ratio)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Cortar preferentemente en salto de línea para no partir frases.
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut > start:
                end = cut + 1
        chunk = text[start:end]
        if start > 0:
            # Recupera el solape respecto al corte anterior ya aplicado en end.
            chunk = text[max(0, start - overlap):end]
            chunks.append(chunk)
        else:
            chunks.append(chunk)
        start = end
    return chunks


def summarize(t: Transcript, txt: str, mode: str, client: LLMClient,
              *, context_chars: int = 24000) -> str:
    """Map-reduce: chunks grandes se resumen parcialmente y se consolidan."""
    system = load_prompt(mode)
    max_chunk = int(context_chars * CONTEXT_FRAC)
    chunks = chunk_by_chars(txt, max_chunk)
    if len(chunks) == 1:
        return client.complete(system, chunks[0])
    partials = [client.complete(system, f"Sección {i + 1}/{len(chunks)}:\n\n{c}")
                for i, c in enumerate(chunks)]
    joined = "\n\n---\n\n".join(f"### Resumen parcial {i + 1}\n{p}" for i, p in enumerate(partials))
    consolidate = ("A continuación hay resúmenes parciales de secciones del mismo audio. "
                   "Consolídalos en UN único informe final coherente, sin duplicar contenido:\n\n" + joined)
    return client.complete(system, consolidate)
