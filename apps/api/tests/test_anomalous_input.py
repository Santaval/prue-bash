"""Entradas anomalas contra la API.

Objetivo declarado del proyecto: que los flujos no produzcan resultados
inesperados al procesar entradas anomalas. Todo lo de aqui debe fallar
limpio con un 4xx, nunca con un 200 raro ni con un 500.
"""

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("name", "body", "reason"),
    [
        ("math", {}, "falta el campo features"),
        ("math", {"features": []}, "lista vacia"),
        ("math", {"features": [1.0, 2.0]}, "sobran entradas"),
        ("climate", {"features": [1.0]}, "faltan entradas"),
        ("math", {"features": "abc"}, "string en vez de lista"),
        ("math", {"features": [None]}, "null como entrada"),
        ("math", {"features": ["x"]}, "texto como entrada"),
        ("math", {"features": [[1.0]]}, "lista anidada"),
        ("math", {"features": {"x": 1.0}}, "objeto en vez de lista"),
        ("math", {"x": 1.0}, "contrato viejo: regresion del cambio rompedor"),
    ],
)
def test_malformed_bodies_are_rejected(client, name, body, reason):
    """Cada cuerpo invalido responde 422 y no una prediccion."""
    response = client.post(f"/predict/{name}", json=body)
    assert response.status_code == 422, f"{reason}: {response.text}"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_never_produce_a_prediction(client, literal):
    """Pydantic acepta NaN e Infinity: el guardia esta en to_batch."""
    response = client.post(
        "/predict/math",
        content=f'{{"features": [{literal}]}}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "prediction" not in response.text


def test_absurdly_long_vector_is_rejected_by_length(client):
    """Se rechaza por cantidad, antes de construir ningun tensor."""
    response = client.post("/predict/math", json={"features": [1.0] * 10_000})
    assert response.status_code == 422
    assert "espera 1 entradas" in response.text


def test_wrong_count_names_the_expected_features(client):
    """El 422 dice cuantas y cuales entradas se esperaban."""
    response = client.post("/predict/climate", json={"features": [0.1, 0.2]})
    assert response.status_code == 422
    assert "temperature_c" in response.json()["detail"]
