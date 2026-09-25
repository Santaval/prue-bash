"""Pruebas del modelo 'climate', el que prueba el camino multi-entrada."""

import numpy as np
import pytest
from prue_model import get_spec, load_model, predict
from trainer.train import train_spec

pytestmark = pytest.mark.slow

SPEC = get_spec("climate")
TRUE_WEIGHTS = [2.0, -1.5, 0.5]
TRUE_BIAS = 0.25


@pytest.fixture(scope="module", name="trained")
def fixture_trained(tmp_path_factory):
    """Entrena 'climate' una vez para todo el modulo."""
    base = tmp_path_factory.mktemp("climate")
    return load_model(train_spec(SPEC, base))


def test_spec_declares_three_named_features():
    """El contrato multi-entrada esta declarado en un solo lugar."""
    assert SPEC.input_features == 3
    assert SPEC.feature_names == ("temperature_c", "humidity", "wind_kph")


def test_dataset_shape():
    """256 muestras de 3 variables."""
    features, targets = SPEC.dataset()
    assert features.shape == (256, 3)
    assert targets.shape == (256,)


def test_dataset_is_deterministic_across_calls():
    """Prueba de que el RNG es propio y no el global que resetea Keras."""
    first_features, first_targets = SPEC.dataset()
    second_features, second_targets = SPEC.dataset()
    np.testing.assert_array_equal(first_features, second_features)
    np.testing.assert_array_equal(first_targets, second_targets)


def test_features_are_normalised():
    """Entradas en [0, 1): con escala real sgd diverge a NaN."""
    features, _ = SPEC.dataset()
    assert features.min() >= 0.0
    assert features.max() < 1.0


def test_trained_weights_recover_the_true_coefficients(trained):
    """El objetivo no tiene ruido, asi que se puede afirmar los coeficientes."""
    kernel, bias = trained.get_weights()
    np.testing.assert_allclose(kernel.ravel(), TRUE_WEIGHTS, atol=0.15)
    np.testing.assert_allclose(bias, [TRUE_BIAS], atol=0.15)


def test_prediction_over_three_features(trained):
    """El camino multi-entrada funciona de punta a punta en la libreria."""
    features = [0.5, 0.5, 0.5]
    expected = float(np.dot(features, TRUE_WEIGHTS) + TRUE_BIAS)
    assert predict(trained, features, SPEC) == pytest.approx(expected, abs=0.15)
