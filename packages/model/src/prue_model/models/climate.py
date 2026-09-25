"""Modelo 'climate': estima la sensacion termica a partir de 3 variables."""

import numpy as np
from keras.models import Sequential

from ..architecture import build_dense_regressor
from ..registry import register

_SAMPLES = 256
_DATA_SEED = 7
# Coeficientes reales que el modelo debe recuperar.
_TRUE_W = np.array([2.0, -1.5, 0.5], dtype=float)
_TRUE_B = 0.25


def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Datos sinteticos deterministas: y = 2t - 1.5h + 0.5w + 0.25.

    Las entradas se generan en [0, 1): sgd con su learning rate por defecto
    diverge a NaN con valores meteorologicos en escala real. Se asume que las
    entradas llegan normalizadas, en el orden de `feature_names`.

    El RNG es propio (default_rng) y no el global de numpy, que
    keras.utils.set_random_seed reinicia: asi los datos son identicos se
    entrene este modelo solo o dentro de --all.
    """
    rng = np.random.default_rng(_DATA_SEED)
    features = rng.uniform(0.0, 1.0, size=(_SAMPLES, 3))
    targets = features @ _TRUE_W + _TRUE_B
    return features, targets


@register(
    name="climate",
    input_features=3,
    feature_names=("temperature_c", "humidity", "wind_kph"),
    dataset=dataset,
    epochs=300,
    seed=42,
    description="Sensacion termica normalizada a partir de 3 variables",
)
def build() -> Sequential:
    """Construye la arquitectura de 'climate'."""
    return build_dense_regressor(3)
