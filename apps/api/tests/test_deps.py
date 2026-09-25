"""Pruebas de carga de artefactos, sin entrenar ni cargar Keras de verdad."""

import pytest
from api import deps
from prue_model import list_names


@pytest.fixture(name="stub_load")
def fixture_stub_load(monkeypatch):
    """Evita cargar Keras: solo interesa que rutas se pidieron."""
    monkeypatch.setattr(deps, "load_model", lambda path: f"modelo:{path.stem}")


def _touch_artifacts(base, names):
    """Crea archivos .keras vacios para los nombres dados."""
    for name in names:
        (base / f"{name}.keras").write_bytes(b"")


def test_refuses_when_models_dir_is_unset(monkeypatch):
    """Se conserva la convencion de negarse a arrancar sin configuracion."""
    monkeypatch.delenv("MODELS_DIR", raising=False)
    with pytest.raises(RuntimeError, match="MODELS_DIR no esta definida"):
        deps.load_all_models()


def test_reports_every_missing_artifact_at_once(monkeypatch, tmp_path):
    """Un solo error completo, no uno por corrida."""
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        deps.load_all_models()
    message = str(excinfo.value)
    for name in list_names():
        assert name in message
    assert "python -m trainer --all" in message


def test_refuses_when_only_one_artifact_is_missing(monkeypatch, tmp_path):
    """Medio servicio es peor que ninguno: tampoco arranca."""
    present, missing = list_names()[0], list_names()[-1]
    _touch_artifacts(tmp_path, [present])
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))

    with pytest.raises(RuntimeError) as excinfo:
        deps.load_all_models()
    message = str(excinfo.value)
    assert missing in message
    assert f"{present}.keras" not in message


@pytest.mark.usefixtures("stub_load")
def test_loads_every_registered_model(monkeypatch, tmp_path):
    """Con todos los artefactos presentes se carga uno por modelo."""
    _touch_artifacts(tmp_path, list_names())
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))

    loaded = deps.load_all_models()
    assert sorted(loaded) == sorted(list_names())
