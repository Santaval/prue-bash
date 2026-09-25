"""Modelo compartido entre el entrenador y la API."""

from .architecture import INPUT_FEATURES, build_model
from .inference import load_model, predict

__all__ = ["INPUT_FEATURES", "build_model", "load_model", "predict"]
