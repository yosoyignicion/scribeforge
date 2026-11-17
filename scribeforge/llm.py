"""Cliente LLM OpenAI-compatible en stdlib (Ollama/LM Studio/Groq). Sin dependencias."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

MAX_RETRIES = 2
BACKOFF_S = 1.0


class LLMError(RuntimeError):
    pass


@dataclass
class Endpoint:
    name: str
    base_url: str
    model: str
    api_key_env: str = ""

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise LLMError(
                    f"Endpoint '{self.name}' requiere la variable de entorno "
                    f"{self.api_key_env} sin valor."
                )
            h["Authorization"] = f"Bearer {key}"
        return h


def chat(endpoint: Endpoint, system: str, user: str, *, timeout: float = 120.0) -> str:
    """Chat completion con reintentos ante 429/5xx y backoff exponencial."""
    body = {
        "model": endpoint.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    last_err: str = ""
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(
            f"{endpoint.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode(),
            headers=endpoint.headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise LLMError(f"Endpoint '{endpoint.name}': respuesta malformada (contenido no textual).")
            return content
        except LLMError:
            raise
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code} desde {endpoint.base_url}"
            if exc.code not in (429, 500, 502, 503, 504):
                raise LLMError(f"Endpoint '{endpoint.name}' respondió {last_err}: {exc.read().decode()[:300]}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = f"endpoint inaccesible o respuesta inválida ({exc})"
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_S * (2 ** attempt))
    raise LLMError(f"Endpoint '{endpoint.name}' falló tras {MAX_RETRIES + 1} intentos: {last_err}")


class LLMClient:
    """Cadena de endpoints: usa el primero disponible, salta al siguiente si falla."""

    def __init__(self, endpoints: list[Endpoint], *, timeout: float = 120.0):
        self.endpoints = endpoints
        self.timeout = timeout
        self.used: str | None = None

    def complete(self, system: str, user: str) -> str:
        if not self.endpoints:
            raise LLMError("No hay endpoints LLM configurados en config.yaml.")
        errors: list[str] = []
        for ep in self.endpoints:
            try:
                out = chat(ep, system, user, timeout=self.timeout)
                self.used = ep.name
                return out
            except LLMError as exc:
                errors.append(str(exc))
        raise LLMError("Todos los endpoints LLM fallaron:\n- " + "\n- ".join(errors))
