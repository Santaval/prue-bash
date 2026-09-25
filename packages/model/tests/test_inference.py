"""Pruebas de validacion e inferencia, sin cargar modelos de Keras."""

import numpy as np
import pytest
from prue_model import (ArtifactNotFoundError, InvalidFeaturesError, get_spec,
                        load_model, predict, to_batch)

MATH = get_spec("math")
CLIMATE = get_spec("climate")


def test_to_batch_shapes_single_feature():
    """El batch queda (1, input_features) y en float32."""
    batch = to_batch([3.0], MATH)
    assert batch.shape == (1, 1)
    assert batch.dtype == np.float32


def test_to_batch_shapes_multiple_features():
    """El camino multi-entrada es el que habilita modelos como climate."""
    assert to_batch([0.1, 0.2, 0.3], CLIMATE).shape == (1, 3)


def test_integers_are_accepted_and_coerced():
    """Un cliente que manda enteros no deberia recibir un error."""
    assert to_batch([1, 2, 3], CLIMATE).dtype == np.float32


@pytest.mark.parametrize("features", [[], [1.0, 2.0], [1.0, 2.0, 3.0]])
def test_wrong_feature_count_is_rejected(features):
    """Entrada anomala: cantidad de entradas distinta a la declarada."""
    with pytest.raises(InvalidFeaturesError) as excinfo:
        to_batch(features, MATH)
    message = str(excinfo.value)
    assert "math" in message and "espera 1 entradas" in message


def test_error_message_names_the_expected_features():
    """El mensaje dice que se esperaba, para que el cliente pueda corregir."""
    with pytest.raises(InvalidFeaturesError, match="temperature_c"):
        to_batch([1.0], CLIMATE)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value):
    """Sin este guardia la API devolveria NaN, que ni siquiera es JSON valido."""
    with pytest.raises(InvalidFeaturesError, match="finitos"):
        to_batch([value], MATH)


@pytest.mark.parametrize("features", [["a"], [None], [{}]])
def test_non_numeric_values_are_rejected(features):
    """Entrada anomala: el TypeError de numpy no debe escaparse."""
    with pytest.raises(InvalidFeaturesError):
        to_batch(features, MATH)


def test_string_is_not_treated_as_a_sequence_of_features():
    """'abc' es una Sequence de largo 3 y colaria como entrada de climate."""
    with pytest.raises(InvalidFeaturesError, match="lista de numeros"):
        to_batch("abc", CLIMATE)


@pytest.mark.parametrize("features", [None, 3.0, {"x": 1}])
def test_non_sequence_inputs_are_rejected(features):
    """Entrada anomala: lo que no es secuencia se rechaza igual."""
    with pytest.raises(InvalidFeaturesError, match="lista de numeros"):
        to_batch(features, MATH)


def test_predict_returns_a_native_float(fake_model):
    """La API serializa el resultado: debe ser float de Python, no np.float32."""
    result = predict(fake_model, [2.0], MATH)
    assert type(result) is float  # pylint: disable=unidiomatic-typecheck


def test_predict_validates_before_touching_the_model(fake_model):
    """La validacion ocurre antes de invocar al modelo."""
    with pytest.raises(InvalidFeaturesError):
        predict(fake_model, [1.0, 2.0], MATH)


def test_load_model_reports_a_missing_artifact_clearly(tmp_path):
    """Mejor un error del dominio que un traceback cripitco de Keras."""
    with pytest.raises(ArtifactNotFoundError, match="No existe el artefacto"):
        load_model(tmp_path / "no-existe.keras")
