"""Pruebas de las rutas HTTP: el flujo de negocio de la API."""

import pytest
from fastapi.testclient import TestClient
from prue_model import list_names

pytestmark = pytest.mark.slow


def test_index_is_a_health_check(client):
    """La raiz sigue sirviendo de health check."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_models_lists_every_registered_model(client):
    """GET /models describe el catalogo, incluido su contrato de entrada."""
    response = client.get("/models")
    assert response.status_code == 200

    body = {entry["name"]: entry for entry in response.json()}
    assert sorted(body) == sorted(list_names())
    assert all(entry["loaded"] for entry in body.values())
    assert body["linear"]["input_features"] == 1
    assert body["plane"]["input_features"] == 3
    assert body["plane"]["feature_names"] == ["a", "b", "c"]


def test_predict_single_feature_model(client):
    """POST /predict/linear sigue aprendiendo y = 2x - 1."""
    response = client.post("/predict/linear", json={"features": [10.0]})
    assert response.status_code == 200

    body = response.json()
    assert body["model"] == "linear"
    assert body["input"] == [10.0]
    assert body["prediction"] == pytest.approx(19.0, abs=0.5)


def test_predict_multi_feature_model(client):
    """La prueba de que el camino multi-entrada funciona sobre HTTP."""
    response = client.post("/predict/plane", json={"features": [0.5, 0.5, 0.5]})
    assert response.status_code == 200

    body = response.json()
    assert body["model"] == "plane"
    assert body["input"] == [0.5, 0.5, 0.5]
    # y = 2a - 1.5b + 0.5c + 0.25 evaluado en (0.5, 0.5, 0.5) = 0.75
    assert body["prediction"] == pytest.approx(0.75, abs=0.2)


def test_unknown_model_is_a_404(client):
    """Entrada anomala: nombre que no existe."""
    response = client.post("/predict/no-existe", json={"features": [1.0]})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "linear" in detail and "plane" in detail


def test_lookup_is_case_sensitive(client):
    """Entrada anomala: 'LINEAR' no es 'linear'."""
    assert client.post("/predict/LINEAR", json={"features": [1.0]}).status_code == 404


def test_unloaded_model_is_a_503():
    """TestClient sin `with` no ejecuta lifespan: nada queda cargado.

    Es a proposito. Cubre la rama 503 y documenta la causa numero uno de
    fallos confusos al escribir pruebas de esta API.
    """
    from api.main import app

    response = TestClient(app).post("/predict/linear", json={"features": [1.0]})
    assert response.status_code == 503
    assert "no esta cargado" in response.json()["detail"]


def test_models_declares_the_output_contract(client):
    """GET /models dice como leer una prediccion antes de pedirla."""
    body = {entry["name"]: entry for entry in client.get("/models").json()}
    for entry in body.values():
        assert entry["task"] == "regression"
        assert entry["class_names"] == []


def test_regression_response_omits_the_classifier_fields(client):
    """Los campos de clasificacion viajan en null cuando no aplican."""
    body = client.post("/predict/linear", json={"features": [10.0]}).json()
    assert body["label"] is None
    assert body["probabilities"] is None
