# AGENTS.md — ScribeForge

## Contexto del repo
- No es un git repo propio: vive dentro del monorepo `~/Agents/`. Los commits desde aquí afectan al repo padre; no hagas commit sin confirmar alcance.
- `spec.md` es la especificación de diseño; `README.md` documenta comportamiento verificado. Si difieren de código, manda el código.

## Entorno (quirk crítico)
- `.venv` creado con `--without-pip` + `get-pip.py` (ensurepip deshabilitado en este sistema): pip está instalado, pero recrear el venv con `python3 -m venv` normal puede fallar. Si hay que recrearlo: `python3 -m venv --without-pip .venv && curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python`.
- Usa siempre `.venv/bin/python`; faster-whisper no está en el Python del sistema.

## Verificación
```bash
.venv/bin/python -m unittest discover -s tests   # 30 smoke tests, sin red ni modelo
.venv/bin/python tests/qa_fixture.py             # 16 checks E2E; requiere ffmpeg real
```
- Los smoke tests son funciones puras (chunker, párrafos, SRT, config, fallback LLM con stubs). No descargan modelo ni tocan red.
- No hay config de lint/mypy/ruff en pyproject (solo constan en `dev` extras): no asumas comandos de lint que no existen.

## Arquitectura y trampas
- Pipeline: `processor.py` (ffmpeg → WAV 16 kHz mono, bloques >2 h con solape) → `transcriber.py` (faster-whisper + VAD + post-proceso anti-alucinación) → `summarizer.py` (chunking map-reduce). Entrypoint: `scribeforge.cli:main`.
- **No añadir `torch`/`torchaudio`**: faster-whisper (CTranslate2) no lo necesita, es decisión de diseño de spec.md.
- No quitar `_shim_pyav()` en `transcriber.py`: esquiva incompatibilidad PyAV ≥ 14 vs faster-whisper ≤ 1.2.1 (`metadata_errors`).
- Caché de transcripción por `(hash input, modelo, idioma)` en `outputs/.cache/` — fuera del workdir efímero a propósito. No moverla.
- Prompts como datos: `scribeforge/prompts/{interview,meeting,lecture}.md`, editables sin tocar código. Están en `package-data` (`.md`): el glob ya cubre ficheros nuevos.
- `config.yaml` en la raíz es opcional: si falta, sigue con defaults avisando una vez (no es error). API key del LLM remoto: env `SCRIBEFORGE_API_KEY`.
- El resumen LLM nunca aborta el pipeline: si no hay endpoint, entrega la transcripción con aviso (exit 0).
