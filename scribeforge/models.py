"""Dataclasses compartidos del pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str = "unknown"
    model: str = ""
    duration: float = 0.0

    @property
    def raw_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())
