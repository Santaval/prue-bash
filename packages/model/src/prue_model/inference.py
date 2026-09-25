"""Carga e inferencia. Unico lugar que valida y arma el batch de entrada.

La validacion vive aqui, no en la API, para que el entrenador y la API
compartan una sola definicion de "entrada valida" y no puedan divergir.
"""

from collections.abc import Sequence
from pathlib import Path

import keras
import numpy as np

from .errors import ArtifactNotFoundError, InvalidFeaturesError
from .registry import ModelSpec


def load_model(path: str | Path) -> keras.Model:
    """Carga un modelo entrenado desde un archivo .keras."""
    target = Path(path)
    if not target.is_file():
        raise ArtifactNotFoundError(f"No existe el artefacto {target}.")
    return keras.models.load_model(target)


def to_batch(features: Sequence[float], spec: ModelSpec) -> np.ndarray:
    """Valida el vector contra el spec y arma el batch (1, input_features)."""
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise InvalidFeaturesError("features debe ser una lista de numeros.")
    if len(features) != spec.input_features:
        raise InvalidFeaturesError(
            f"El modelo '{spec.name}' espera {spec.input_features} entradas "
            f"({', '.join(spec.feature_names)}); se recibieron {len(features)}."
        )
    try:
        batch = np.asarray(features, dtype=np.float32).reshape(1, spec.input_features)
    except (TypeError, ValueError) as exc:
        raise InvalidFeaturesError(f"Entradas no numericas: {exc}") from exc
    # Pydantic acepta NaN e Infinity para un float, asi que sin este guardia
    # la API devolveria una prediccion NaN que ni siquiera es JSON valido.
    if not np.all(np.isfinite(batch)):
        raise InvalidFeaturesError("Las entradas deben ser numeros finitos.")
    return batch


def predict(model: keras.Model, features: Sequence[float], spec: ModelSpec) -> float:
    """Predice para un vector de entradas y devuelve un float nativo de Python."""
    batch = to_batch(features, spec)
    return float(model.predict(batch, verbose=0)[0][0])
