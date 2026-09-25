"""Dependencias externas de la API."""

import os

import keras
from prue_model import load_model


def load_configured_model() -> keras.Model:
    """Carga el modelo desde la ruta configurada en MODEL_PATH."""
    path = os.environ.get("MODEL_PATH")
    if not path:
        raise RuntimeError("MODEL_PATH no esta definida. Corre la tarea del trainer primero.")
    return load_model(path)
