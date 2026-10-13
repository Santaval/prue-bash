"""Pruebas de la CLI del entrenador."""

from pathlib import Path

import pytest
from prue_model import list_names
from trainer import cli


@pytest.fixture(name="recorded")
def fixture_recorded(monkeypatch):
    """Sustituye el entrenamiento real y anota que specs se pidieron."""
    calls: list[tuple[str, Path]] = []

    def _fake_train_spec(spec, base_dir):
        calls.append((spec.name, base_dir))
        return base_dir / f"{spec.name}.keras"

    monkeypatch.setattr(cli, "train_spec", _fake_train_spec)
    return calls


def test_single_model_trains_only_that_one(recorded, tmp_path):
    """--model entrena exactamente uno."""
    assert cli.main(["--model", "linear", "--models-dir", str(tmp_path)]) == 0
    assert [name for name, _ in recorded] == ["linear"]


def test_all_covers_every_registered_model(recorded, tmp_path):
    """--all no puede saltarse un modelo nuevo del registro."""
    assert cli.main(["--all", "--models-dir", str(tmp_path)]) == 0
    assert [name for name, _ in recorded] == list(list_names())


def test_all_with_an_empty_registry_says_so(recorded, registry_snapshot, capsys, tmp_path):
    """Un clon recien clonado no trae modelos: --all lo explica, no calla."""
    registry_snapshot.restore({})
    assert cli.main(["--all", "--models-dir", str(tmp_path)]) == 0
    assert not recorded
    assert "No hay modelos registrados" in capsys.readouterr().out


def test_unknown_model_exits_with_code_two(recorded, capsys):
    """Entrada anomala: nombre inexistente."""
    assert cli.main(["--model", "no-existe"]) == 2
    assert not recorded
    assert "no existe" in capsys.readouterr().err


def test_model_and_all_together_are_rejected(tmp_path):
    """Son mutuamente excluyentes."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--model", "linear", "--all", "--models-dir", str(tmp_path)])
    assert excinfo.value.code == 2


def test_a_target_is_required():
    """Sin --model ni --all no hay nada que entrenar."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


def test_models_dir_falls_back_to_the_environment(recorded, monkeypatch, tmp_path):
    """Sin --models-dir se usa MODELS_DIR."""
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    assert cli.main(["--model", "linear"]) == 0
    assert recorded[0][1] == tmp_path


@pytest.mark.slow
def test_real_run_writes_the_artifact(tmp_path):
    """Corrida real de punta a punta de la CLI."""
    assert cli.main(["--model", "linear", "--models-dir", str(tmp_path)]) == 0
    assert (tmp_path / "linear.keras").is_file()
