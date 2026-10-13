"""Pruebas de resolucion de MODELS_DIR y rutas de artefactos."""

from pathlib import Path

import pytest
from prue_model import DEFAULT_MODELS_DIR, artifact_path, get_spec, models_dir


def test_required_and_unset_refuses(monkeypatch):
    """La API depende de esto para negarse a arrancar sin configuracion."""
    monkeypatch.delenv("MODELS_DIR", raising=False)
    with pytest.raises(RuntimeError, match="MODELS_DIR no esta definida"):
        models_dir(required=True)


@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_blank_values_count_as_unset(monkeypatch, raw):
    """Entrada anomala: una variable en blanco no es configuracion valida."""
    monkeypatch.setenv("MODELS_DIR", raw)
    with pytest.raises(RuntimeError):
        models_dir(required=True)


def test_optional_falls_back_to_the_default(monkeypatch):
    """El entrenador puede correr en un clon recien clonado."""
    monkeypatch.delenv("MODELS_DIR", raising=False)
    assert models_dir(required=False) == Path(DEFAULT_MODELS_DIR)


def test_configured_value_is_used(monkeypatch, tmp_path):
    """Lo que diga el entorno manda."""
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    assert models_dir() == tmp_path


def test_artifact_path_is_name_dot_keras(tmp_path):
    """Comparacion como Path, nunca como string: separadores de Windows."""
    spec = get_spec("plane")
    assert artifact_path(spec, tmp_path) == tmp_path / "plane.keras"


def test_artifact_path_uses_the_environment_when_no_base(monkeypatch, tmp_path):
    """Sin base explicita cae en MODELS_DIR."""
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    assert artifact_path(get_spec("linear")) == tmp_path / "linear.keras"
