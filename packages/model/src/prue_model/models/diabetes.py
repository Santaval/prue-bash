"""Modelo 'diabetes': regresion logistica del riesgo de diabetes (M07)."""

from pathlib import Path

import keras
import numpy as np
from keras.layers import Dense, Normalization

from ..registry import register

_TRAIN = Path("data/processed/diabetes_train.csv")
FEATURES = ("gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level")
# Para un binario la sigmoide da la probabilidad del SEGUNDO nombre.
CLASS_NAMES = ("no_diabetes", "diabetes")


def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Lee el split de entrenamiento: 8 columnas de entrada y diabetes (0/1)."""
    data = np.loadtxt(_TRAIN, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, :-1], data[:, -1].astype(int)


@register(
    name="diabetes",
    input_features=len(FEATURES),
    feature_names=FEATURES,
    dataset=dataset,
    epochs=3,  # ~77k filas: converge en 3; mas epocas solo alargan la suite
    description="Regresion logistica: riesgo de diabetes (valores reales)",
    class_names=CLASS_NAMES,
)
def build() -> keras.Model:
    """Normaliza -> Dense(1, sigmoid). Sin capas ocultas: regresion logistica.

    La normalizacion vive dentro del modelo (media y varianza quedan en el
    .keras), asi que la API recibe los valores crudos, sin escalar.
    """
    features, _ = dataset()
    normalize = Normalization()
    normalize.adapt(features)

    model = keras.Sequential([
        keras.Input(shape=(len(FEATURES),)),
        normalize,
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model
