"""Pruebas de validacion e inferencia, sin cargar modelos de Keras."""

import numpy as np
import pytest
from prue_model import (ArtifactNotFoundError, InvalidFeaturesError, ModelSpec,
                        OutputShapeError, get_spec, interpret, load_model,
                        predict, to_batch)

LINEAR = get_spec("linear")
PLANE = get_spec("plane")


def _classifier_spec(class_names: tuple[str, ...]) -> ModelSpec:
    """Spec desechable: no se registra, solo describe la salida a interpretar.

    La infraestructura soporta clasificacion pero no hay ningun clasificador
    registrado, asi que las pruebas traen el suyo.
    """
    return ModelSpec(
        name="tmp",
        input_features=1,
        feature_names=("a",),
        build=lambda: None,
        dataset=lambda: (np.zeros((2, 1)), np.zeros(2)),
        epochs=1,
        class_names=class_names,
    )


BINARY = _classifier_spec(("ham", "spam"))
MULTICLASS = _classifier_spec(("baja", "media", "alta"))


def test_to_batch_shapes_single_feature():
    """El batch queda (1, input_features) y en float32."""
    batch = to_batch([3.0], LINEAR)
    assert batch.shape == (1, 1)
    assert batch.dtype == np.float32


def test_to_batch_shapes_multiple_features():
    """El camino multi-entrada es el que habilita modelos como plane."""
    assert to_batch([0.1, 0.2, 0.3], PLANE).shape == (1, 3)


def test_integers_are_accepted_and_coerced():
    """Un cliente que manda enteros no deberia recibir un error."""
    assert to_batch([1, 2, 3], PLANE).dtype == np.float32


@pytest.mark.parametrize("features", [[], [1.0, 2.0], [1.0, 2.0, 3.0]])
def test_wrong_feature_count_is_rejected(features):
    """Entrada anomala: cantidad de entradas distinta a la declarada."""
    with pytest.raises(InvalidFeaturesError) as excinfo:
        to_batch(features, LINEAR)
    message = str(excinfo.value)
    assert "linear" in message and "espera 1 entradas" in message


def test_error_message_names_the_expected_features():
    """El mensaje dice que se esperaba, para que el cliente pueda corregir."""
    with pytest.raises(InvalidFeaturesError, match=r"\(a, b, c\)"):
        to_batch([1.0], PLANE)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value):
    """Sin este guardia la API devolveria NaN, que ni siquiera es JSON valido."""
    with pytest.raises(InvalidFeaturesError, match="finitos"):
        to_batch([value], LINEAR)


@pytest.mark.parametrize("features", [["a"], [None], [{}]])
def test_non_numeric_values_are_rejected(features):
    """Entrada anomala: el TypeError de numpy no debe escaparse."""
    with pytest.raises(InvalidFeaturesError):
        to_batch(features, LINEAR)


def test_string_is_not_treated_as_a_sequence_of_features():
    """'abc' es una Sequence de largo 3 y colaria como entrada de plane."""
    with pytest.raises(InvalidFeaturesError, match="lista de numeros"):
        to_batch("abc", PLANE)


@pytest.mark.parametrize("features", [None, 3.0, {"x": 1}])
def test_non_sequence_inputs_are_rejected(features):
    """Entrada anomala: lo que no es secuencia se rechaza igual."""
    with pytest.raises(InvalidFeaturesError, match="lista de numeros"):
        to_batch(features, LINEAR)


def test_predict_returns_a_native_float(fake_model):
    """La API serializa el resultado: debe ser float de Python, no np.float32."""
    result = predict(fake_model, [2.0], LINEAR)
    assert type(result.value) is float  # pylint: disable=unidiomatic-typecheck


def test_regression_leaves_the_classifier_fields_empty(fake_model):
    """Una regresion no inventa etiquetas ni probabilidades."""
    result = predict(fake_model, [2.0], LINEAR)
    assert result.label is None
    assert result.probabilities is None


def test_predict_validates_before_touching_the_model(fake_model):
    """La validacion ocurre antes de invocar al modelo."""
    with pytest.raises(InvalidFeaturesError):
        predict(fake_model, [1.0, 2.0], LINEAR)


def test_load_model_reports_a_missing_artifact_clearly(tmp_path):
    """Mejor un error del dominio que un traceback cripitco de Keras."""
    with pytest.raises(ArtifactNotFoundError, match="No existe el artefacto"):
        load_model(tmp_path / "no-existe.keras")


def test_binary_prediction_reports_label_and_probabilities(fake_binary_model):
    """Una sigmoide de 0.8 es la clase positiva, con ambas probabilidades."""
    result = predict(fake_binary_model, [1.0], BINARY)
    assert result.value == pytest.approx(0.8)
    assert result.label == "spam"
    assert result.probabilities == pytest.approx({"ham": 0.2, "spam": 0.8})


@pytest.mark.parametrize(
    ("score", "label"),
    [(0.49, "ham"), (0.5, "spam"), (0.51, "spam")],
)
def test_binary_label_flips_at_the_threshold(score, label):
    """El umbral es 0.5 y 0.5 exacto cae del lado positivo."""
    assert interpret(np.array([[score]]), BINARY).label == label


def test_multiclass_picks_the_argmax_label(fake_multiclass_model):
    """La clase ganadora es la de mayor probabilidad, nombrada por el spec."""
    result = predict(fake_multiclass_model, [1.0], MULTICLASS)
    assert result.label == "media"
    assert result.value == 1.0  # el indice de la clase ganadora
    assert result.probabilities == pytest.approx(
        {"baja": 0.1, "media": 0.7, "alta": 0.2}
    )


@pytest.mark.parametrize("spec", [LINEAR, BINARY, MULTICLASS], ids=lambda s: s.task)
def test_every_number_is_a_native_python_float(spec):
    """Regresion: la API serializa esto y np.float32 no es JSON."""
    raw = np.zeros((1, spec.output_units), dtype=np.float32)
    result = interpret(raw, spec)
    assert type(result.value) is float  # pylint: disable=unidiomatic-typecheck
    for probability in (result.probabilities or {}).values():
        assert type(probability) is float  # pylint: disable=unidiomatic-typecheck


@pytest.mark.parametrize(
    ("raw", "spec"),
    [
        (np.array([[0.1, 0.9]]), BINARY),      # artefacto de 2 unidades vs spec de 1
        (np.array([[0.5]]), MULTICLASS),       # artefacto viejo vs spec ampliado
    ],
)
def test_output_wider_than_the_spec_is_rejected(raw, spec):
    """Un artefacto desfasado falla explicito, no con un IndexError."""
    with pytest.raises(OutputShapeError, match="Reentrena"):
        interpret(raw, spec)
