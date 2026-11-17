"""Configuración: defaults, carga perezosa de YAML y validación que lista TODAS las violaciones."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULTS: dict = {
    "model": {
        "size": "small",
        "device": "auto",
        "compute_type": None,
        "language": "auto",
        "vram_gpu_gb": 6,
    },
    "summary": {
        "mode": "meeting",
        "context_chars": 24000,
        "chunk_overlap": 0.1,
        "timeout_s": 120,
        "retries": 2,
    },
    "llm": {
        "endpoints": [
            {"name": "local-ollama", "base_url": "http://localhost:11434/v1",
             "model": "qwen3:4b", "api_key_env": ""},
            {"name": "groq", "base_url": "https://api.groq.com/openai/v1",
             "model": "llama-3.3-70b-versatile", "api_key_env": "SCRIBEFORGE_API_KEY"},
        ],
    },
    "media": {"segment_threshold_h": 2, "block_minutes": 30, "block_overlap_s": 2.0},
}

VALID_MODES = {"interview", "meeting", "lecture"}


def validate(cfg: dict) -> list[str]:
    """Devuelve TODAS las violaciones (una línea por campo), nunca falla en cascada."""
    errs: list[str] = []

    def section(key):
        val = cfg.get(key, DEFAULTS[key])
        return val if isinstance(val, dict) else None

    model = section("model")
    if model:
        if model.get("size") not in {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}:
            errs.append(f"model.size: valor '{model.get('size')}' inválido "
                        f"(esperados: tiny|base|small|medium|large-v3)")
        if model.get("device") not in {"auto", "cpu", "cuda"}:
            errs.append(f"model.device: '{model.get('device')}' inválido (auto|cpu|cuda)")
        if not isinstance(model.get("vram_gpu_gb", 6), int):
            errs.append("model.vram_gpu_gb: debe ser un entero (GB de VRAM disponible)")

    summary = section("summary")
    if summary:
        if summary.get("mode") not in VALID_MODES:
            errs.append(f"summary.mode: '{summary.get('mode')}' inválido "
                        f"(esperados: {'|'.join(sorted(VALID_MODES))})")
        ctx = summary.get("context_chars", 24000)
        if not isinstance(ctx, int) or not (1000 <= ctx <= 2_000_000):
            errs.append(f"summary.context_chars: '{ctx}' fuera de rango [1000, 2000000]")
        ov = summary.get("chunk_overlap", 0.1)
        if not isinstance(ov, (int, float)) or not (0 <= ov < 0.5):
            errs.append(f"summary.chunk_overlap: '{ov}' debe ser un número en [0, 0.5)")
        if not isinstance(summary.get("timeout_s", 120), (int, float)) or summary.get("timeout_s", 120) <= 0:
            errs.append(f"summary.timeout_s: '{summary.get('timeout_s')}' debe ser un número > 0")

    media = section("media")
    if media:
        ov_s = media.get("block_overlap_s", 2.0)
        bm = media.get("block_minutes", 30)
        if isinstance(bm, (int, float)) and isinstance(ov_s, (int, float)) and ov_s >= bm * 60:
            errs.append(f"media.block_overlap_s: solape {ov_s}s debe ser < block_minutes*60 ({bm * 60}s)")

    llm = section("llm")
    if llm:
        eps = llm.get("endpoints")
        if not isinstance(eps, list) or not eps:
            errs.append("llm.endpoints: debe ser una lista con al menos un endpoint")
        else:
            for i, ep in enumerate(eps):
                url = ep.get("base_url", "") if isinstance(ep, dict) else ""
                if not url.startswith(("http://", "https://")):
                    errs.append(f"llm.endpoints[{i}].base_url: '{url}' no es una URL http(s) válida")
                if not (ep.get("model") if isinstance(ep, dict) else None):
                    errs.append(f"llm.endpoints[{i}].model: falta o está vacío")
    return errs


def load(path: Path | None = None, *, warn=print) -> dict:
    """Carga config.yaml si existe (yaml perezoso). Sin archivo: defaults + aviso una vez."""
    cfg = {**{k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in DEFAULTS.items()},
           "llm": {"endpoints": [dict(e) for e in DEFAULTS["llm"]["endpoints"]]}}
    target = path or Path("config.yaml")
    if not target.exists():
        warn(f"[scribeforge] config.yaml no encontrado en {target}; usando defaults documentados.")
        return cfg
    try:
        import yaml
    except ModuleNotFoundError:
        raise RuntimeError(
            "config.yaml existe pero falta PyYAML, y no puede interpretar sin él. "
            "Instálalo con `pip install pyyaml` o borra config.yaml para usar defaults."
        )
    user_cfg = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(user_cfg, dict):
        raise RuntimeError(f"config.yaml: contenido raíz debe ser un mapeo, no {type(user_cfg).__name__}.")
    for k, v in user_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


REPO_DEFAULT_ENV_KEY = "SCRIBEFORGE_API_KEY"


def drop_endpoints_without_access(cfg: dict) -> list[str]:
    """Nombres de endpoints cuyo api_key_env está fijado pero sin valor → avisables."""
    missing = []
    for ep in cfg.get("llm", {}).get("endpoints", []):
        env = ep.get("api_key_env") or ""
        if env and not os.environ.get(env):
            missing.append(ep.get("name") or "<sin-nombre>")
    return missing
