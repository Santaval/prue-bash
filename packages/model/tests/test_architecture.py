"""Pruebas de la arquitectura parametrizada."""

import keras
import numpy as np
import pytest
from prue_model import build_dense_classifier, build_dense_regressor


@pytest.mark.parametrize("features", [1, 3, 10])
def test_shapes_follow_the_argument(features):
    """La forma de entrada la decide el llamador, no una constante global."""
    model = build_dense_regressor(features)
    assert model.input_shape == (None, features)
    assert model.output_shape == (None, 1)


def test_compiled_with_expected_optimizer_and_loss():
    """Se conserva sgd / mean_squared_error como configuracion por defecto."""
    model = build_dense_regressor(1)
    assert model.optimizer.__class__.__name__ == "SGD"
    assert model.loss == "mean_squared_error"


def test_optimizer_is_configurable():
    """Un modelo con datos sin normalizar puede necesitar otro optimizador."""
    model = build_dense_regressor(3, optimizer="adam")
    assert model.optimizer.__class__.__name__ == "Adam"


@pytest.mark.parametrize("features", [0, -1])
def test_rejects_non_positive_feature_counts(features):
    """Entrada anomala: un modelo sin entradas no tiene sentido."""
    with pytest.raises(ValueError, match="input_features"):
        build_dense_regressor(features)


def test_building_has_no_side_effects():
    """Importar y construir no entrena: cada llamada da un objeto nuevo."""
    first = build_dense_regressor(1)
    second = build_dense_regressor(1)
    assert first is not second


def test_binary_head_is_a_single_sigmoid():
    """Dos clases se resuelven con una unidad: la probabilidad de la positiva."""
    model = build_dense_classifier(4, classes=2)
    assert model.input_shape == (None, 4)
    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"
    assert model.loss == "binary_crossentropy"


@pytest.mark.parametrize("classes", [3, 5])
def test_multiclass_head_is_a_softmax_per_class(classes):
    """Tres o mas clases dan una unidad por clase."""
    model = build_dense_classifier(2, classes=classes)
    assert model.output_shape == (None, classes)
    assert model.layers[-1].activation.__name__ == "softmax"
    assert model.loss == "sparse_categorical_crossentropy"


@pytest.mark.parametrize("classes", [1, 0, -1])
def test_rejects_fewer_than_two_classes(classes):
    """Entrada anomala: un clasificador de una sola clase no decide nada."""
    with pytest.raises(ValueError, match="classes debe ser >= 2"):
        build_dense_classifier(1, classes=classes)


def test_classifier_rejects_non_positive_feature_counts():
    """El mismo guardia de entrada que el regresor."""
    with pytest.raises(ValueError, match="input_features"):
        build_dense_classifier(0, classes=2)


def test_classifier_defaults_to_adam():
    """sgd por defecto avanza demasiado lento sobre crossentropy."""
    model = build_dense_classifier(1, classes=2)
    assert model.optimizer.__class__.__name__ == "Adam"


def test_same_seed_yields_same_initial_weights():
    """Ancla a nivel unitario de la historia de reproducibilidad."""
    keras.utils.set_random_seed(42)
    first = build_dense_regressor(1).get_weights()
    keras.utils.set_random_seed(42)
    second = build_dense_regressor(1).get_weights()
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
