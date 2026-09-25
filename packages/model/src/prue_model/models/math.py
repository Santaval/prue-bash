"""Modelo 'math': aprende la relacion lineal y = 2x - 1."""

import numpy as np
from keras.models import Sequential

from ..architecture import build_dense_regressor
from ..registry import register

# Relacion a aprender: y = 2x - 1
_X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
_Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Datos de entrenamiento. Devuelve copias para que nadie los mute."""
    return _X.copy(), _Y.copy()


@register(
    name="math",
    input_features=1,
    feature_names=("x",),
    dataset=dataset,
    epochs=500,
    seed=42,
    description="Regresion lineal y = 2x - 1",
)
def build() -> Sequential:
    """Construye la arquitectura de 'math'."""
    return build_dense_regressor(1)
