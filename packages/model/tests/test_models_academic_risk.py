"""Pruebas del modelo 'academic_risk' (clasificacion multiclase)."""

from pathlib import Path

import numpy as np
import pytest
from keras.layers import Dense
from prue_model import (InvalidFeaturesError, artifact_path, get_spec,
                        load_model, predict, to_batch)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPEC = get_spec("academic_risk")
_TEST = Path("data/processed/academic_risk_test.csv")
_LOW_RISK = [95.0, 9.0, 20.0, 0.0, 95.0, 10.0, 0.0]
_HIGH_RISK = [55.0, 5.0, 5.0, 5.0, 40.0, 0.0, 1.0]


def _split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, :-1], data[:, -1].astype(int)


# --- Contrato y datos (rapidos) ----------------------------------------------

def test_spec_declares_the_input_contract():
    """Siete entradas en este orden y tres clases: lo que exige la API."""
    assert SPEC.input_features == 7
    assert SPEC.feature_names == (
        "attendance_pct", "previous_gpa", "study_hours_weekly",
        "failed_courses_history", "assignments_on_time_pct",
        "forum_participation_count", "is_working",
    )
    assert SPEC.task == "multiclass"
    assert SPEC.class_names == ("0", "1", "2")


def test_dataset_is_clean_and_deterministic():
    """Sin NaN ni negativos, objetivo 0/1/2 con las tres clases presentes."""
    first, targets = SPEC.dataset()
    first = np.asarray(first)
    targets = np.asarray(targets)
    second, _ = SPEC.dataset()
    assert first.shape[1] == 7 and first.shape[0] == targets.shape[0] > 0
    assert np.isfinite(first).all() and (first >= 0).all()
    assert set(np.unique(targets)) == {0, 1, 2}
    np.testing.assert_array_equal(first, np.asarray(second))


def test_architecture_is_a_multiclass_classifier():
    """Tres unidades Dense con softmax producen una distribucion sobre 0/1/2."""
    dense = [layer for layer in SPEC.build().layers if isinstance(layer, Dense)]
    assert len(dense) == 1
    assert dense[0].units == 3
    assert dense[0].activation.__name__ == "softmax"


@pytest.mark.parametrize("features", [
    _LOW_RISK[:-1],                                      # falta una entrada
    _LOW_RISK + [99.0],                                  # sobra una
    [95.0, 9.0, 20.0, 0.0, float("nan"), 10.0, 0.0],   # NaN
    [95.0, 9.0, float("inf"), 0.0, 95.0, 10.0, 0.0],  # infinito
])
def test_rejects_anomalous_inputs(features):
    """Entradas fuera del contrato se rechazan antes de predecir."""
    with pytest.raises(InvalidFeaturesError):
        to_batch(features, SPEC)


# --- Modelo entrenado (lentos) -----------------------------------------------

@pytest.fixture(scope="module", name="model")
def fixture_model(trained_models_dir):
    """Reutiliza el entrenamiento de sesion de conftest: no entrena dos veces."""
    return load_model(artifact_path(SPEC, trained_models_dir))


@pytest.fixture(scope="module", name="test_split")
def fixture_test_split():
    """El 20% que el modelo nunca vio durante el entrenamiento."""
    return _split(_TEST)


@pytest.mark.slow
def test_generalizes_to_unseen_data(model, test_split):
    """El modelo supera un minimo en exactitud global y entre las tres clases."""
    features, targets = test_split
    predicted = np.argmax(model.predict(features, verbose=0), axis=1)
    assert accuracy_score(targets, predicted) >= 0.65
    assert balanced_accuracy_score(targets, predicted) >= 0.55


@pytest.mark.slow
def test_matches_sklearn_logistic_regression(model, test_split):
    """Keras debe alcanzar una exactitud cercana al clasificador lineal base."""
    features, targets = test_split
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    baseline.fit(*SPEC.dataset())
    keras_accuracy = accuracy_score(targets, np.argmax(
        model.predict(features, verbose=0), axis=1
    ))
    sklearn_accuracy = accuracy_score(targets, baseline.predict(features))
    assert keras_accuracy == pytest.approx(sklearn_accuracy, abs=0.12)


@pytest.mark.slow
def test_example_predictions_have_the_expected_labels(model):
    """Casos claros via predict(): bajo riesgo es 0 y alto riesgo es 2."""
    low = predict(model, _LOW_RISK, SPEC)
    high = predict(model, _HIGH_RISK, SPEC)
    assert low.label == "0"
    assert high.label == "2"
    assert sum(low.probabilities.values()) == pytest.approx(1.0)
    assert sum(high.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.slow
def test_more_failed_courses_increases_the_high_risk(model):
    """Con todo lo demas igual, mas cursos reprobados suben la clase 2."""
    low = predict(model, _LOW_RISK, SPEC)
    more_failed = predict(model,
                          _LOW_RISK[:3] + [3.0] + _LOW_RISK[4:], SPEC)
    assert more_failed.probabilities["2"] > low.probabilities["2"]
