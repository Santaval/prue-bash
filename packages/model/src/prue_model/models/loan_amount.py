"""Modelo 'loan_amount': regresion lineal multiple del monto de un prestamo."""

from pathlib import Path

import keras
import numpy as np
from keras.layers import Dense, Normalization, Rescaling

from ..registry import register

_TRAIN = Path("data/processed/loan_train.csv")
FEATURES = ("applicant_income", "coapplicant_income", "credit_score",
            "years_employed", "owns_property", "loan_term_months")


def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Lee el split de entrenamiento: 6 columnas de entrada y loan_amount."""
    data = np.loadtxt(_TRAIN, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, :-1], data[:, -1]


@register(
    name="loan_amount",
    input_features=len(FEATURES),
    feature_names=FEATURES,
    dataset=dataset,
    epochs=50,
    description="Regresion lineal multiple: monto del prestamo (valores reales)",
)
def build() -> keras.Model:
    """Normaliza -> Dense(1) -> desnormaliza. Todo sigue siendo lineal."""
    features, targets = dataset()
    normalize = Normalization()
    normalize.adapt(features)  # media y varianza quedan guardadas en el .keras

    model = keras.Sequential([
        keras.Input(shape=(len(FEATURES),)),
        normalize,
        Dense(1),
        Rescaling(scale=float(targets.std()), offset=float(targets.mean())),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error")
    return model
