"""Pruebas del modelo 'diabetes' (regresion logistica)."""

from pathlib import Path

import numpy as np
import pytest
from keras.layers import Dense
from prue_model import (InvalidFeaturesError, artifact_path, get_spec,
                        load_model, predict, to_batch)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPEC = get_spec("diabetes")
_TEST = Path("data/processed/diabetes_test.csv")
# gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c, glucosa
_HEALTHY = [0.0, 30.0, 0.0, 0.0, 0.0, 22.0, 5.0, 90.0]
_AT_RISK = [1.0, 65.0, 1.0, 1.0, 1.0, 35.0, 8.0, 240.0]


def _split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, :-1], data[:, -1].astype(int)


# --- Contrato y datos (rapidos) ----------------------------------------------

def test_spec_declares_the_input_contract():
    """Ocho entradas en este orden y dos clases: lo que la API exige."""
    assert SPEC.input_features == 8
    assert SPEC.feature_names == (
        "gender", "age", "hypertension", "heart_disease",
        "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level",
    )
    assert SPEC.task == "binary"
    assert SPEC.class_names == ("no_diabetes", "diabetes")


def test_dataset_is_clean_and_deterministic():
    """Sin NaN ni negativos, objetivo 0/1 con ambas clases, mismo resultado."""
    first, targets = SPEC.dataset()
    second, _ = SPEC.dataset()
    assert first.shape[1] == 8 and first.shape[0] == targets.shape[0] > 0
    assert np.isfinite(first).all() and (first >= 0).all()
    assert set(np.unique(targets)) == {0, 1}
    np.testing.assert_array_equal(first, second)


def test_architecture_is_a_logistic_regression():
    """Una sola Dense de 1 unidad con sigmoide: p = sigmoid(w.x + b)."""
    dense = [layer for layer in SPEC.build().layers if isinstance(layer, Dense)]
    assert len(dense) == 1
    assert dense[0].units == 1
    assert dense[0].activation.__name__ == "sigmoid"


@pytest.mark.parametrize("features", [
    _HEALTHY[:-1],                                       # falta una entrada
    _HEALTHY + [1.0],                                    # sobra una
    [0.0, 30.0, 0.0, 0.0, 0.0, float("nan"), 5.0, 90.0], # NaN
    [0.0, 30.0, 0.0, 0.0, 0.0, 22.0, float("inf"), 90.0],# infinito
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


@pytest.fixture(scope="module", name="test_split")
def fixture_test_split():
    """El 20% que el modelo nunca vio."""
    return _split(_TEST)


@pytest.mark.slow
def test_generalizes_to_unseen_data(model, test_split):
    """AUC alto y detecta a la mayoria... sin conformarse con la exactitud.

    Con 91% de negativos, decir siempre 'no' ya da ~91% de exactitud; por eso
    se mide AUC y recall de la clase positiva.
    """
    features, targets = test_split
    probabilities = model.predict(features, verbose=0).ravel()
    assert roc_auc_score(targets, probabilities) >= 0.95
    assert recall_score(targets, probabilities >= 0.5) >= 0.55


@pytest.mark.slow
def test_matches_sklearn_logistic_regression(model, test_split):
    """Keras debe llegar practicamente al mismo AUC que sklearn."""
    features, targets = test_split
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    baseline.fit(*SPEC.dataset())
    keras_auc = roc_auc_score(targets, model.predict(features, verbose=0).ravel())
    exact_auc = roc_auc_score(targets, baseline.predict_proba(features)[:, 1])
    assert keras_auc == pytest.approx(exact_auc, abs=0.01)


@pytest.mark.slow
def test_example_predictions_have_the_expected_labels(model):
    """Casos claros via predict(), igual que los recibe la API."""
    healthy = predict(model, _HEALTHY, SPEC)
    at_risk = predict(model, _AT_RISK, SPEC)
    assert healthy.label == "no_diabetes"
    assert at_risk.label == "diabetes"
    assert sum(at_risk.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.slow
def test_higher_hba1c_increases_the_risk(model):
    """Regla clinica: con todo lo demas igual, mas HbA1c sube la probabilidad."""
    low = predict(model, _HEALTHY, SPEC).value
    high = predict(model, _HEALTHY[:6] + [7.5] + _HEALTHY[7:], SPEC).value
    assert high > low
