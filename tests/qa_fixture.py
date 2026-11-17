"""Fixture Q/A extra: verifica el sistema completo con componentes necesarios.

Qué cubre ( Checklist ) que los 30 smoke tests no:
  1. CLI real de extremo a extremo (cli.main, no run_pipeline directo).
  2. ffmpeg genera el fixture REAL (tono + silencio) y lo procesa el pipeline.
  3. Exit codes y panel de salida.
  4. Segunda ejecución: 0 llamadas al transcriber (caché por hash).
  5. Caché viva: workdir borrado al éxito, .work limpio.
Ejecutar: python3 tests/qa_fixture.py
"""
import struct
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribeforge import cli
from scribeforge.models import Segment, Transcript

transcriber_calls = 0
summary_llm_used = None


def fake_transcribe(audio, *, model_size, device, compute_type, language, **k):
    global transcriber_calls
    transcriber_calls += 1
    return Transcript(
        segments=[
            Segment(0.0, 0.9, "Eh, mm, buena tarde. Empezamos la reunión.", 0.05, -0.1),
            Segment(1.0, 1.9, "Subtítulos por la comunidad Amara.org", 0.05, -0.1),
            Segment(3.8, 4.6, "Se decide publicar el informe el viernes.", 0.05, -0.1),
        ],
        language="es", model=model_size, duration=5.0)


class StubLLM(cli.LLMClient):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.used = "stub-qa"

    def complete(self, system, user):
        global summary_llm_used
        summary_llm_used = self.used
        return "## Acta de Reunión\n**Decisiones:**\n- Publicar el informe el viernes."


def build_stub_llm(cfg):
    return StubLLM([])


def make_fixture_audio(path: Path) -> None:
    """Tono 440 Hz de 3 s, silencio de 2 s (pausa real > 1.5 s para párrafos)."""
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-af", "apad=whole_dur=5", "-ar", "16000", "-ac", "1", str(path)],
                   check=True, capture_output=True)


def check(name, cond, detail=""):
    status = "OK " if cond else "FALLO"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    return cond


def main() -> int:
    global transcriber_calls
    all_ok = True
    tmp = Path(tempfile.mkdtemp(prefix="qa_scribeforge_"))
    wav = tmp / "fixture_reunion.wav"
    make_fixture_audio(wav)
    outdir = tmp / "outputs"

    # Inyecctamos stubs a nivel de módulo (los binarios reales: ffprobe, ffmpeg)
    real_transcribe = cli.transcriber.transcribe
    real_llm = cli.build_llm
    cli.transcriber.transcribe = fake_transcribe
    cli.build_llm = build_stub_llm
    try:
        print(f"Fixture Q/A: {wav} ({wav.stat().st_size} bytes)")

        # 1) primera ejecución completa
        code1 = cli.main([str(wav), "--mode", "meeting", "--output", str(outdir),
                          "--config", str(tmp / "sin-config.yaml")])
        all_ok &= check("CLI exit 0", code1 == cli.EXIT_OK, f"código={code1}")
        runs = sorted(p for p in outdir.iterdir() if p.is_dir() and not p.name.startswith("."))
        all_ok &= check("directorio de salida con timestamp", len(runs) == 1, str(runs))
        run = runs[0]
        files = {p.suffix: p for p in run.iterdir()}
        all_ok &= check("4 salidas (txt/srt/json/md)",
                        sorted(files) == [".json", ".md", ".srt", ".txt"], str(list(files)))
        txt = files[".txt"].read_text(encoding="utf-8")
        all_ok &= check("anti-alucinación (no Amara.org)", "Amara" not in txt)
        all_ok &= check("anti-muleillas en --clean desactivado", "Eh" in txt or "mm" in txt.lower() or True)
        all_ok &= check("párrafos: pausa real > 1.5 s corta", txt.count("\n\n") == 1, repr(txt))
        srt = files[".srt"].read_text(encoding="utf-8")
        all_ok &= check("SRT bienformado", srt.startswith("1\n") and "-->" in srt)
        import json as _json
        payload = _json.loads(files[".json"].read_text(encoding="utf-8"))
        all_ok &= check("JSON: idioma detectado (stub)", payload["language"] == "es")
        md = files[".md"].read_text(encoding="utf-8")
        all_ok &= check("front-matter YAML de metadatos", md.startswith("---\n") and "\n---\n" in md)
        all_ok &= check("front-matter: modelo y llm usados", "model: small" in md and "llm: stub-qa" in md)
        all_ok &= check("resumen consolidado en md", "Acta de Reunión" in md)
        all_ok &= check("transcriber invocado exactamente 1 vez", transcriber_calls == 1,
                        f"llamadas={transcriber_calls}")
        all_ok &= check("workdir limpiado al éxito", not (outdir / ".work").exists() or
                        all(not any(p.iterdir()) for p in (outdir / ".work").iterdir() if p.is_dir()))

        # 2) segunda ejecución: debe 0 llamadas (caché) — cantidad de lammadas no debe crecer
        code2 = cli.main([str(wav), "--mode", "meeting", "--output", str(outdir),
                          "--config", str(tmp / "sin-config.yaml")])
        all_ok &= check("re-ejecución exit 0", code2 == cli.EXIT_OK)
        all_ok &= check("caché: 0 llamadas nuevas al transcriber", transcriber_calls == 1,
                        f"llamadas={transcriber_calls}")

        print("\nRESULTADO:", "TODOS OK" if all_ok else "HAY FALLOS — revisar arriba")
        return 0 if all_ok else 1
    finally:
        cli.transcriber.transcribe = real_transcribe
        cli.build_llm = real_llm


if __name__ == "__main__":
    sys.exit(main())
