"""Carga e inferencia. Unico lugar que valida y arma el batch de entrada.

La validacion vive aqui, no en la API, para que el entrenador y la API
compartan una sola definicion de "entrada valida" y no puedan divergir.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np

from .errors import (ArtifactNotFoundError, InvalidFeaturesError,
                     OutputShapeError)
from .registry import ModelSpec


@dataclass(frozen=True)
class Prediction:
    """Resultado de una prediccion, ya interpretado segun el spec.

    `value` siempre esta: es el valor estimado en una regresion, la
    probabilidad de la clase positiva en un binario y el indice de la clase
    ganadora en un multiclase. `label` y `probabilities` solo los llenan los
    clasificadores.
    """

    value: float
    label: str | None = None
    probabilities: dict[str, float] | None = None


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


def interpret(raw: np.ndarray, spec: ModelSpec) -> Prediction:
    """Traduce la salida cruda del modelo a la forma que declara el spec.

    Todos los numeros salen como float nativo de Python: la API los serializa
    y un np.float32 no es JSON.
    """
    row = np.asarray(raw).reshape(-1)
    if row.size != spec.output_units:
        raise OutputShapeError(
            f"El modelo '{spec.name}' devolvio {row.size} salidas pero su spec "
            f"declara {spec.output_units}. Reentrena el artefacto."
        )

    if spec.task == "regression":
        return Prediction(value=float(row[0]))

    if spec.task == "binary":
        # El umbral es 0.5 porque la cabeza es una sigmoide sobre la clase que
        # ocupa la segunda posicion de class_names.
        positive = float(row[0])
        negative_name, positive_name = spec.class_names
        return Prediction(
            value=positive,
            label=positive_name if positive >= 0.5 else negative_name,
            probabilities={negative_name: 1.0 - positive, positive_name: positive},
        )

    index = int(np.argmax(row))
    return Prediction(
        value=float(index),
        label=spec.class_names[index],
        probabilities=dict(zip(spec.class_names, (float(item) for item in row))),
    )


def predict(
    model: keras.Model, features: Sequence[float], spec: ModelSpec
) -> Prediction:
    """Predice para un vector de entradas y devuelve la prediccion interpretada."""
    batch = to_batch(features, spec)
    return interpret(model.predict(batch, verbose=0), spec)
