"""Constructores de arquitecturas reutilizables, sin efectos secundarios."""

import keras
from keras.layers import Dense
from keras.models import Sequential


def build_dense_regressor(
    input_features: int,
    *,
    units: int = 1,
    optimizer: str = "sgd",
    loss: str = "mean_squared_error",
) -> Sequential:
    """Construye y compila un regresor denso sin entrenarlo.

    `optimizer` es configurable porque sgd con su learning rate por defecto
    diverge si las entradas no estan normalizadas; un modelo con datos reales
    puede necesitar Adam sin cambiar de arquitectura.
    """
    if input_features < 1:
        raise ValueError("input_features debe ser >= 1.")
    model = Sequential([keras.Input(shape=(input_features,)), Dense(units=units)])
    model.compile(optimizer=optimizer, loss=loss)
    return model
