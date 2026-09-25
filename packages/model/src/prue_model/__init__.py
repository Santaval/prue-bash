"""Modelo compartido entre el entrenador y la API."""

# Importar `models` tiene el efecto de llenar el registro: cada archivo de
# ese paquete se da de alta solo al importarse.
from . import models  # pylint: disable=unused-import
from .architecture import build_dense_regressor
from .artifacts import DEFAULT_MODELS_DIR, artifact_path, models_dir
from .errors import (ArtifactNotFoundError, DuplicateModelError,
                     InvalidFeaturesError, ModelNotFoundError, PrueModelError)
from .inference import load_model, predict, to_batch
from .registry import ModelSpec, get_spec, list_names, list_specs, register

__all__ = [
    "DEFAULT_MODELS_DIR",
    "ArtifactNotFoundError",
    "DuplicateModelError",
    "InvalidFeaturesError",
    "ModelNotFoundError",
    "ModelSpec",
    "PrueModelError",
    "artifact_path",
    "build_dense_regressor",
    "get_spec",
    "list_names",
    "list_specs",
    "load_model",
    "models",
    "models_dir",
    "predict",
    "register",
    "to_batch",
]
