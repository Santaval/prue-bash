"""Carga e inferencia. Unico lugar que conoce la forma del batch de entrada."""

import keras
import numpy as np

from .architecture import INPUT_FEATURES


def load_model(path: str) -> keras.Model:
    """Carga un modelo entrenado desde un archivo .keras."""
    return keras.models.load_model(path)


def predict(model: keras.Model, x: float) -> float:
    """Predice para un unico valor y devuelve un float nativo de Python."""
    batch = np.array([[x]], dtype=np.float32).reshape(1, INPUT_FEATURES)
    return float(model.predict(batch, verbose=0)[0][0])
