"""Pruebas de CORS: la UI sirve desde otro origen y debe poder llamar a la API."""

import pytest
from fastapi.testclient import TestClient

from api.main import _cors_origins, app


@pytest.fixture(name="cors_client")
def fixture_cors_client(monkeypatch):
    """Cliente sin cargar modelos; el middleware usa los origenes por defecto."""
    monkeypatch.setenv("API_SKIP_MODEL_LOADING", "true")
    with TestClient(app) as test_client:
        yield test_client


def _preflight(client, origin):
    return client.options(
        "/models",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )


def test_default_origins_include_local_ui(monkeypatch):
    """Sin variable, se permiten los dos origenes locales de la UI."""
    monkeypatch.delenv("API_CORS_ORIGINS", raising=False)
    assert _cors_origins() == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_origins_are_read_from_env(monkeypatch):
    """La variable se separa por comas y se ignoran espacios y vacios."""
    monkeypatch.setenv("API_CORS_ORIGINS", " https://a.example , ,https://b.example")
    assert _cors_origins() == ["https://a.example", "https://b.example"]


def test_preflight_allows_listed_origin(cors_client):
    """Un origen permitido recibe access-control-allow-origin."""
    response = _preflight(cors_client, "http://localhost:5173")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_preflight_rejects_unlisted_origin(cors_client):
    """Un origen no listado no recibe access-control-allow-origin."""
    response = _preflight(cors_client, "http://evil.example")
    assert "access-control-allow-origin" not in response.headers
