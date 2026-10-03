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


def build_dense_classifier(
    input_features: int,
    *,
    classes: int,
    optimizer: str = "adam",
) -> Sequential:
    """Construye y compila un clasificador denso sin entrenarlo.

    La cabeza la decide `classes`: dos clases se resuelven con una sola unidad
    sigmoide (la probabilidad de la clase positiva), y tres o mas con una
    softmax de `classes` unidades. Esa es la misma correspondencia que
    `ModelSpec.output_units` declara del lado del registro.

    A diferencia del regresor, el optimizador por defecto es adam: sgd con su
    learning rate por defecto avanza demasiado lento sobre crossentropy.
    """
    if input_features < 1:
        raise ValueError("input_features debe ser >= 1.")
    if classes < 2:
        raise ValueError("classes debe ser >= 2: un clasificador de una sola "
                         "clase no decide nada.")

    if classes == 2:
        units, activation, loss = 1, "sigmoid", "binary_crossentropy"
    else:
        units, activation = classes, "softmax"
        loss = "sparse_categorical_crossentropy"

    model = Sequential(
        [keras.Input(shape=(input_features,)), Dense(units=units, activation=activation)]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
