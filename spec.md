# Especificación Técnica de Proyecto: `ScribeForge-Local`

> **Definición del Producto:** Utilidad local en Python orientada a creadores, periodistas e investigadores para procesar archivos de audio y vídeo de cualquier duración, transcribirlos con precisión mediante IA optimizada y generar resúmenes estructurados sin depender de APIs de pago ni sufrir muros de transcripción por minutos.

## 1. Arquitectura del Sistema y Stack Tecnológico

- **Core de Lenguaje:** Python 3.10+ (tipado estricto con `mypy`/`pyright` en CI local).
- **Procesamiento Multimedia:** `ffmpeg` (binario nativo del sistema, invocado por subproceso).
- **Motor de Transcripción:** `faster-whisper` (CTranslate2). **No depende de torch** — no incluir `torch`/`torchaudio` en dependencias. GPU opcional vía cuDNN/CTranslate2.
- **Motor de Resumen:** Cliente OpenAI-compatible apuntando a endpoints locales (Ollama en `http://localhost:11434/v1`, LM Studio) o remotos (Groq/Mistral API). Configurable 100% por `config.yaml` + variables de entorno (`SCRIBEFORGE_API_KEY`).
- **Interfaz:** CLI con `rich` (progreso, tablas, avisos). Panel web (Streamlit) queda **fuera de la v1** — backlog.

## 2. Estructura de Directorios del Proyecto

```
scribeforge/
├── pyproject.toml           # Packaging + entry point `scribeforge` (console script)
├── config.example.yaml      # Plantilla de configuración (el usuario copia a config.yaml)
├── scribeforge/
│   ├── __init__.py
│   ├── cli.py               # Orquestación, flags y output rich (antes main.py)
│   ├── processor.py         # FFmpeg: validación, extracción, segmentación
│   ├── transcriber.py       # Wrapper faster-whisper + caché de resultados
│   ├── summarizer.py        # Chunking, prompts por modo, map-reduce, reintentos
│   ├── llm.py               # Cliente LLM agnóstico (local/remoto), timeouts y retry
│   └── models.py            # Dataclasses compartidos (Segment, Transcript, Summary)
├── outputs/                 # Transcripciones y resúmenes (git-ignorado por defecto)
└── tests/
    ├── fixtures/            # Audio sintético corto (\< 10 s) generado en CI con ffmpeg
    ├── test_processor.py
    ├── test_chunker.py
    └── test_cli.py
```

## 3. Especificación de Capacidades por Fase

### Fase 1: Ingesta, Validación y Preprocesamiento (`processor.py`)

- Validar binario `ffmpeg` al arrancar (`ffmpeg -version`, falla con mensaje accionable si falta).
- Aceptar `.mp4`, `.mkv`, `.mov`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.opus`, `.ogg`, `.webm`.
- Extracción a **WAV 16 kHz mono PCM s16** con `-hide_banner -loglevel error` (silencia spam de copys):
  ```
  ffmpeg -i input -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
  ```
- **Workdir por archivo:** todo intermedio vive en `outputs/.work/\<hash-sha256-8del-input>/` para que sea reanudable y limpiable sin afectar otros jobs. Limpieza al terminar con éxito; el flag `--keep-work` lo conserva para depurar.
- Duración > 2 h: segmentar por silencio con `silencedetect` en bloques de ~30 min con 2 s de solape (evita cortar palabras). La transcripción re-ensambla y desplaza timestamps por el offset de bloque.

### Fase 2: Transcripción (`transcriber.py`)

- Selección de modelo por hardware (tabla en `config.yaml`, auto-detección con fallback):

  | Memoria disponible | Modelo | `compute_type` |
  |---|---|---|
  | GPU ≥ 6 GB VRAM | `large-v3` | `float16` |
  | GPU < 6 GB o CPU fuerte | `small` | `int8` |
  | CPU modesta | `base` | `int8` |
- Overrides: `--model`, `--device`, `--compute-type`, `--language` (default: auto, guarda el idioma detectado en el JSON).
- Inferencia con `vad_filter=True` (recorta silencios, menos alucinaciones en posibles gaps) y `beam_size=5`.
- **Batch explícito** (v1.1): `BatchedInferencePipeline` con `batch_size` configurable — hasta ~8x en GPU.
- **Post-procesado automático** (aquí vive la diferencia entre transcript mediocre y premium):
  - Filtrar segmentos con `no_speech_prob > 0.6` y `avg_logprob < -1.0` (alucinaciones típicas: bucles de "Subtítulos por...").
  - Desduplicar repeticiones consecutivas idénticas de Whisper.
  - Reconstrucción de párrafos por pausas > 1.5 s y fin de frase, en vez de volcar segmento a segmento.
  - Opcional `--clean`: limpieza de muletillas con heurísticas (sin LLM) previa al `.txt`.
- Salidas por transcripción (en `outputs/\<nombre-input>-YYYYMMDD-HHMM/`):
  1. `.txt` plano con párrafos limpios.
  2. `.srt` con timestamps de segmento.
  3. `.json` con segmentos crudos (start, end, text, no-speech-prob) — material para post-proceso y clave de **caché**.
- **Resumen/caché:** la transcripción se cachea por `(hash_input, modelo, idioma)`. Reprocesar solo cuesta la fase de resumen. `--no-cache` fuerza re-transcripción.

### Fase 3: Resumen y Estructuración (`summarizer.py` + `llm.py`)

- Chunking por caracteres (proxy robusto sin tokenizador extra) al ~70% del contexto configurado, con **solape de 10%** y mezcla map-reduce: resumen parcial por chunk → consolidación final. Chunks ≤ 1 ventana: un solo pase.
- Prompts por modo (`--mode`), versionados en `scribeforge/prompts/<modo>.md` para poder editarlos sin tocar código:
  - `interview`: preguntas clave, respuestas destacadas con timestamp, conclusiones.
  - `meeting`: acta, decisiones, action items con responsable y fecha implícita si se menciona.
  - `lecture`: índice de temas, resumen ejecutivo, preguntas de repaso.
- Salida de resumen en `.md` con **front-matter de metadatos** ( YAML entre `---`): archivo original, duración, modelo usado, idioma detectado, timestamps de cada sección, fecha de proceso y versión de ScribeForge. Cualquier salida es auto-documentada y grepeable.
- **Fiabilidad del LLM:**
  - Timeouts y 2 reintentos con backoff exponencial ante 429/5xx.
  - Fallback de endpoint: si el primario no responde, prueba el secundario definido en `config.yaml`.
  - Si la clave de un endpoint remoto no está, salta directamente al fallback local con aviso, no aborta.
- Salida de resumen en markdown en el mismo directorio de outputs.

### Fase 4: CLI y Entrega (`cli.py`)

- Instalable: `pip install -e .` expone `scribeforge`. Ejemplos:
  ```
  scribeforge reunion_equipo.mp4 --mode meeting
  scribeforge entrevista.m4a --mode interview --model medium --language es
  scribeforge --dry-run charla.mp4        # muestra plan (modelo, segs, coste estimado) sin ejecutar
  ```
- Progreso con `rich`: fases numeradas, barra por transcripción segment a segment, ETA. Al terminar: panel resumen (duración del audio, tiempo de proceso, ratio x, archivos generados con rutas clicables).
- Errores: códigos de salida distintos (input inválido / sin ffmpeg / LLM inaccesible / cancelado). Ctrl+C: cancela limpio, conserva lo ya escrito, borra workdir salvo `--keep-work`.
- `--version`, `--config <path>`, `--dry-run`.
- Mensajes de error siempre accionables: qué falló, por qué, y el comando o configuración que lo arregla. Nada de tracebacks crudos al usuario (traceback completo solo con `--debug`).

## 4. Validación de Configuración

- `config.yaml` validado al cargar (pydantic): tipos, rangos (ej. `chunk_overlap` < `chunk_size`), endpoints URL bien formados.
- Config ilegal → un solo error que lista **todas** las violaciones con la ruta del campo y el valor esperado, no falla en cascada campo a campo.
- Si `config.yaml` no existe: usa defaults documentados y avisa una sola vez. Nunca crash por ausencia de config.

## 5. Dependencias (`pyproject.toml` / `requirements.txt`)

```
faster-whisper>=1.0.0
rich>=13.0.0
PyYAML>=6.0
openai>=1.30.0        # cliente compat OpenAI para Ollama/LM Studio/Groq
pydantic>=2.0.0       # validación de config.yaml con errores accionables
```

Dev: `pytest`, `mypy`. `requests` solo si se decide no usar el SDK `openai` (el SDK cubre retry estándar y es menos código propio).

## 6. Calidad Premium (puertas y pulido)

- **Tooling obligatorio en dev:** `ruff` (lint+format), `mypy --strict` en el paquete, `pytest` con cobertura mínima en `summarizer` (chunker + merge son lógica pura: 90%+). Pre-commit local, sin drama.
- **Tests con golden files:** fixture de audio sintético (tonos + silencio, generado con ffmpeg, < 10 s, cero red) y su transcripción esperada cacheada. El chunker, el merge map-reduce y el post-procesado se testean como funciones puras con transcripciones de ejemplo — no requieren modelo ni red.
- **Documentación como parte del producto:**
  - `README.md` con GIF/demo del CLI, tabla de requisitos y quickstart de 30 s.
  - `--help` generada con rich y ejemplos reales por flag (no descripciones abstractas).
  - `docs/benchmarks.md`: tabla real medida en máquina del autor (minuto de audio → tiempo por modelo/hardware). El spec no promete x4/x8: el repo publica lo medido.
- **Privacidad por diseño:** cero telemetría, cero red en transcripción, el `.json` con metadatos nunca sale del disco. Se documenta explícitamente en el README (sy selling point para periodistas).

## 7. Decisiones de Diseño (no negociables)

1. **Sin torch.** faster-whisper = CTranslate2; instalar torch es agregar 2.5 GB sin ganancia.
2. **Local-first:** todo funciona sin red salvo el resumen con endpoint remoto; la transcripción nunca toca la red.
3. **Reanudable y observable:** workdir persistente por hash, caché de transcripción, códigos de salida claros.
4. **Prompts como datos:** markdown versionado, no strings embebidas.
5. **Backlog explícito, no scope-creep:** diarización de hablantes (pyannote), panel web, exportación VTT/ASS, watch-folder. Ninguno entra en v1.
