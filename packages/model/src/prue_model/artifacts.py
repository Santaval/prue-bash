"""Resolucion de rutas de artefactos a partir de MODELS_DIR.

Lo comparten el entrenador y la API para que no puedan discrepar sobre donde
viven los artefactos, por la misma razon que comparten el contrato de forma.
"""

import os
from pathlib import Path

from .registry import ModelSpec

DEFAULT_MODELS_DIR = "apps/trainer/dist"


def models_dir(required: bool = True) -> Path:
    """Devuelve el directorio de artefactos configurado en MODELS_DIR.

    La API lo pide con `required=True` y se niega a arrancar si falta. El
    entrenador usa `required=False` para que un clon nuevo pueda entrenar sin
    configurar nada. La ruta por defecto es relativa al directorio de trabajo:
    los comandos se corren desde la raiz del repositorio.
    """
    raw = os.environ.get("MODELS_DIR", "").strip()
    if not raw:
        if required:
            raise RuntimeError(
                "MODELS_DIR no esta definida. Corre la tarea del trainer primero."
            )
        raw = DEFAULT_MODELS_DIR
    return Path(raw)


def artifact_path(spec: ModelSpec, base: Path | None = None) -> Path:
    """Ruta del artefacto de un modelo: <MODELS_DIR>/<name>.keras."""
    return (base if base is not None else models_dir()) / f"{spec.name}.keras"
