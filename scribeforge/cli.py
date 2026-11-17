"""Fase 4: CLI y orquestación del pipeline completo."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from . import config as cfgmod
from . import processor, summarizer, transcriber
from .llm import Endpoint, LLMClient, LLMError
from .models import Transcript

__version__ = "0.1.0"

EXIT_OK, EXIT_BAD_INPUT, EXIT_NO_FFMPEG, EXIT_LLM, EXIT_CANCELLED = 0, 2, 3, 4, 130


def _plain(msg: str) -> None:
    print(msg)


def front_matter(input_path: Path, t: Transcript, mode: str, llm_used: str | None,
                 elapsed: float) -> str:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["---",
             f"input: {input_path.name}",
             f"duration_s: {t.duration:.1f}",
             f"model: {t.model}",
             f"language: {t.language}",
             f"mode: {mode}",
             f"llm: {llm_used or 'no resumen'}",
             f"elapsed_s: {elapsed:.1f}",
             f"scribeforge: {__version__}",
             f"processed_at: {now}",
             "---", ""]
    return "\n".join(lines)


def build_llm(cfg: dict) -> LLMClient:
    eps = [Endpoint(name=e.get("name", f"ep{i}"), base_url=e["base_url"],
                    model=e["model"], api_key_env=e.get("api_key_env", ""))
           for i, e in enumerate(cfg["llm"]["endpoints"])]
    return LLMClient(eps, timeout=cfg["summary"]["timeout_s"])


def run_pipeline(input_path: Path, *, mode: str, output_dir: Path, model: str,
                 device: str, compute_type: str | None, language: str, clean: bool,
                 no_cache: bool, keep_work: bool, transcribe_fn=None,
                 cfg: dict | None = None, silent: bool = False) -> dict[str, Path]:
    """Extract → transcribe (cacheada) → salidas → resumen. Devuelve rutas generadas."""
    transcribe_fn = transcribe_fn or transcriber.transcribe
    log = _plain if not silent else (lambda *_: None)
    cfg = cfg or cfgmod.DEFAULTS
    processor.check_ffmpeg()
    processor.validate_input(input_path)

    duration = processor.probe_duration(input_path)
    inp_hash = processor.sha256_file(input_path)[:16]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    outdir = output_dir / f"{input_path.stem}-{stamp}"
    workdir = output_dir / ".work" / inp_hash
    made: dict[str, Path] = {}
    ok = False
    start = time.monotonic()
    try:
        log(f"[1/4] Extrayendo audio ({duration / 60:.1f} min)…")
        audio = processor.extract_audio(input_path, workdir)

        cache_dir = output_dir / ".cache"
        cache = cache_dir / f"{inp_hash}.json"
        cache_key = json.dumps({"h": inp_hash, "m": model, "l": language})
        t: Transcript | None = None
        if not no_cache and cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("key") == cache_key:
                t = Transcript(
                    segments=[transcriber.Segment(**s) for s in cached["segments"]],
                    language=cached["language"], model=cached["model"],
                    duration=cached["duration"])
                log("[2/4] Transcripción en caché (hash + modelo coinciden).")
        if t is None:
            log(f"[2/4] Transcribiendo con modelo '{model}'…")
            t = transcribe_fn(audio, model_size=model, device=device,
                              compute_type=compute_type, language=language)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"key": cache_key, "language": t.language,
                                         "model": t.model, "duration": t.duration,
                                         "segments": [vars(s) for s in t.segments]}), encoding="utf-8")

        log("[3/4] Escribiendo transcripción (txt/srt/json)…")
        made.update(transcriber.save_outputs(t, outdir, clean=clean))

        log(f"[4/4] Resumen modo '{mode}'…")
        client = build_llm(cfg)
        txt = made["txt"].read_text(encoding="utf-8")
        if txt.strip():
            try:
                summary = summarizer.summarize(t, txt, mode, client,
                                               context_chars=cfg["summary"]["context_chars"])
                md = outdir / "resumen.md"
                md.write_text(front_matter(input_path, t, mode, client.used,
                                           time.monotonic() - start) + summary, encoding="utf-8")
                made["md"] = md
            except LLMError as exc:
                log(f"[4/4] AVISO: resumen omitido — {exc}")
        ok = True
        return made
    finally:
        if ok and not keep_work:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
        elif not ok:
            log("[workdir] conservado a causa de un fallo (reanudará si se reintenta).")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scribeforge",
        description="Transcribe audio/vídeo local con faster-whisper y genera resúmenes LLM.",
        epilog="Ejemplo: scribeforge reunion.mp4 --mode meeting")
    p.add_argument("input", nargs="?", type=Path, help="archivo de audio o vídeo")
    p.add_argument("--mode", default="meeting", choices=summarizer.MODES,
                   help="perfil de resumen (default: meeting)")
    p.add_argument("--model", default="auto",
                   help="tiny|base|small|medium|large-v3 (default: auto → el de config)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute-type", default=None, help="int8|float16|…")
    p.add_argument("--language", default="auto",
                   help="código ISO-639-1 o 'auto' para detectar")
    p.add_argument("--output", type=Path, default=Path("outputs"), help="directorio raíz de salida")
    p.add_argument("--clean", action="store_true", help="limpia muletillas con heurísticas (sin LLM)")
    p.add_argument("--no-cache", action="store_true", help="fuerza re-transcripción")
    p.add_argument("--keep-work", action="store_true", help="conserva archivos intermedios para depurar")
    p.add_argument("--dry-run", action="store_true", help="muestra el plan sin ejecutar nada")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--debug", action="store_true", help="tracebacks completos en error")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)

    cfg = cfgmod.load(args.config, warn=_plain)
    violations = cfgmod.validate(cfg)
    if violations:
        _plain("config.yaml inválido:\n" + "\n".join(f"  - {e}" for e in violations))
        return EXIT_BAD_INPUT

    model = args.model if args.model != "auto" else cfg["model"]["size"]
    if args.input is None:
        p.error("se requiere un archivo de entrada")
    if not args.input.exists():
        _plain(f"ERROR: no existe {args.input}")
        return EXIT_BAD_INPUT
    if args.input.suffix.lower() not in processor.SUPPORTED_EXTS:
        _plain(f"ERROR: extensión '{args.input.suffix}' no soportada. "
               f"Válidas: {', '.join(sorted(processor.SUPPORTED_EXTS))}")
        return EXIT_BAD_INPUT

    if args.dry_run:
        duration = (args.input.exists() and processor.probe_duration(args.input)) or None
        dur = f"{duration / 60:.1f} min" if duration else "desconocida"
        _plain(f"Plan:\n  entrada: {args.input} ({dur})\n  modelo: {model} en {args.device}\n"
               f"  modo: {args.mode}\n  salida: {args.output}/")
        return EXIT_OK

    try:
        start = time.monotonic()
        made = run_pipeline(args.input, mode=args.mode, output_dir=args.output, model=model,
                            device=args.device, compute_type=args.compute_type,
                            language=args.language if args.language != "auto" else None,
                            clean=args.clean, no_cache=args.no_cache, keep_work=args.keep_work,
                            cfg=cfg)
        elapsed = time.monotonic() - start
        _plain("Listo. Archivos generados:")
        for k, v in made.items():
            _plain(f"  - [{k}] {v}")
        _plain(f"Tiempo total: {elapsed:.1f}s")
        return EXIT_OK
    except KeyboardInterrupt:
        _plain("Cancelado por el usuario. Se conservan los archivos ya escritos.")
        return EXIT_CANCELLED
    except processor.ProcessorError as exc:
        _plain(f"ERROR: {exc}")
        return EXIT_NO_FFMPEG if "ffmpeg no encontrado" in str(exc) else EXIT_BAD_INPUT
    except LLMError as exc:
        _plain(f"ERROR LLM: {exc}")
        return EXIT_LLM
    except Exception as exc:  # último recurso accionable, nunca traceback crudo salvo --debug
        if args.debug:
            raise
        _plain(f"ERROR inesperado: {exc!r}. Reintenta con --debug para el traceback completo.")
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())
