"""Pruebas del modelo 'loan_amount' (regresion lineal multiple)."""

from pathlib import Path

import numpy as np
import pytest
from keras.layers import Dense
from prue_model import (InvalidFeaturesError, artifact_path, get_spec,
                        load_model, predict, to_batch)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

SPEC = get_spec("loan_amount")
_TEST = Path("data/processed/loan_test.csv")
_EXAMPLE = [5000.0, 1000.0, 650.0, 10.0, 1.0, 36.0]


def _split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, :-1], data[:, -1]


# --- Contrato y datos (rapidos) ----------------------------------------------

def test_spec_declares_the_input_contract():
    """Seis entradas en este orden: es lo que la API exige a los clientes."""
    assert SPEC.input_features == 6
    assert SPEC.feature_names == (
        "applicant_income", "coapplicant_income", "credit_score",
        "years_employed", "owns_property", "loan_term_months",
    )
    assert SPEC.task == "regression"


def test_dataset_is_clean_and_deterministic():
    """Sin NaN ni negativos, y los mismos datos en cada llamada."""
    first, targets = SPEC.dataset()
    second, _ = SPEC.dataset()
    assert first.shape[1] == 6 and first.shape[0] == targets.shape[0] > 0
    assert np.isfinite(first).all() and np.isfinite(targets).all()
    assert (first >= 0).all() and (targets > 0).all()
    np.testing.assert_array_equal(first, second)


def test_architecture_is_a_linear_regression():
    """Una sola Dense de 1 unidad y sin activacion: y = w.x + b."""
    dense = [layer for layer in SPEC.build().layers if isinstance(layer, Dense)]
    assert len(dense) == 1
    assert dense[0].units == 1
    assert dense[0].activation.__name__ == "linear"


@pytest.mark.parametrize("features", [
    [5000.0, 1000.0, 650.0, 10.0, 1.0],               # falta una entrada
    [5000.0, 1000.0, 650.0, 10.0, 1.0, 36.0, 99.0],   # sobra una
    [5000.0, 1000.0, float("nan"), 10.0, 1.0, 36.0],  # NaN
    [5000.0, 1000.0, 650.0, float("inf"), 1.0, 36.0], # infinito
])
def test_rejects_anomalous_inputs(features):
    """Entradas que no cumplen el contrato se rechazan antes de predecir."""
    with pytest.raises(InvalidFeaturesError):
        to_batch(features, SPEC)


# --- Modelo entrenado (lentos) -----------------------------------------------

@pytest.fixture(scope="module", name="model")
def fixture_model(trained_models_dir):
    """Reutiliza el entrenamiento de sesion de conftest: no entrena dos veces."""
    return load_model(artifact_path(SPEC, trained_models_dir))


@pytest.fixture(scope="module", name="baseline")
def fixture_baseline():
    """Solucion exacta por minimos cuadrados: el techo de un modelo lineal."""
    return LinearRegression().fit(*SPEC.dataset())


@pytest.mark.slow
def test_generalizes_to_unseen_data(model):
    """R^2 sobre el 20% que el modelo nunca vio."""
    features, targets = _split(_TEST)
    predicted = model.predict(features, verbose=0).ravel()
    assert r2_score(targets, predicted) >= 0.85


@pytest.mark.slow
def test_converges_to_the_least_squares_solution(model, baseline):
    """Keras debe llegar practicamente al mismo R^2 que la solucion exacta."""
    features, targets = _split(_TEST)
    keras_r2 = r2_score(targets, model.predict(features, verbose=0).ravel())
    exact_r2 = r2_score(targets, baseline.predict(features))
    assert keras_r2 == pytest.approx(exact_r2, abs=0.01)


@pytest.mark.slow
def test_example_prediction_matches_the_baseline(model, baseline):
    """Un caso concreto via predict(), igual que lo recibe la API."""
    expected = baseline.predict(np.array([_EXAMPLE]))[0]
    assert predict(model, _EXAMPLE, SPEC).value == pytest.approx(expected, rel=0.02)


@pytest.mark.slow
def test_owning_property_increases_the_loan(model):
    """Regla de negocio: con todo lo demas igual, tener propiedad sube el monto."""
    without = predict(model, [5000.0, 1000.0, 650.0, 10.0, 0.0, 36.0], SPEC).value
    with_property = predict(model, _EXAMPLE, SPEC).value
    assert with_property > without
