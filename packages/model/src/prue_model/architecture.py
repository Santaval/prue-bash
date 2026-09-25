"""Definicion de la arquitectura del modelo, sin efectos secundarios."""

import keras
from keras.layers import Dense
from keras.models import Sequential

INPUT_FEATURES = 1


def build_model() -> Sequential:
    """Construye y compila el modelo sin entrenarlo."""
    model = Sequential([keras.Input(shape=(INPUT_FEATURES,)), Dense(units=1)])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    return model
