# Adding a model, end to end

This tutorial walks through adding a model from an empty file to a working
HTTP prediction, with tests. It takes about ten minutes, most of it waiting
for training.

You will build **`exam_score`**: it predicts a student's exam score from how
many hours they studied and how many hours they slept. Then, as a variant,
you will turn it into **`exam_pass`**, a classifier that answers *pass* or
*fail*.

Every command and output below was run against this repository. For the
reasoning behind each piece, see [models.md](models.md).

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Decide the contracts](#2-decide-the-contracts)
3. [Create the model file](#3-create-the-model-file)
4. [Check that it registered](#4-check-that-it-registered)
5. [Train it](#5-train-it)
6. [Serve it and predict](#6-serve-it-and-predict)
7. [Write its tests](#7-write-its-tests)
8. [Run the whole suite](#8-run-the-whole-suite)
9. [Commit](#9-commit)
10. [Variant: a classifier](#10-variant-a-classifier)
11. [Troubleshooting](#11-troubleshooting)
12. [Checklist](#12-checklist)

## 1. Prerequisites

Finish the *Environment Setup* in the [main README](../README.md): virtual
environment active, `pip install -r requirements.txt` done, and `PYTHONPATH`
exported. Run every command **from the repository root**.

```bash
# Linux/macOS
export PYTHONPATH="packages/model/src:apps/api/src:apps/trainer/src"
# Windows (PowerShell)
$env:PYTHONPATH = "packages/model/src;apps/api/src;apps/trainer/src"
```

## 2. Decide the contracts

Before writing code, answer three questions. They become the model's
`ModelSpec`, and the trainer, the API and the tests all read from it.

| Question | `exam_score` | Spec field |
|----------|--------------|------------|
| What is it called? | `exam_score` (lowercase, no spaces: it goes in a URL) | `name` |
| What goes in, in what order? | `hours_studied`, `hours_slept` | `input_features=2`, `feature_names` |
| What comes out? | A number (the score) → regression | `class_names=()` (the default) |

One more decision: **scaling**. The default regressor trains with `sgd`,
which diverges to `NaN` on raw values like "8 hours" or "70 points". So the
model works on normalized values:

- inputs: `hours / 12` (so 8 hours → `0.6667`)
- output: `score / 100` (so `0.70` means 70 points)

Clients must send normalized inputs; the library does not rescale for them.
Saying so in `description` makes it visible in `GET /models`.

## 3. Create the model file

Create `packages/model/src/prue_model/models/exam_score.py`:

```python
"""Modelo 'exam_score': estima la nota de un examen a partir de horas de estudio y sueno."""

import numpy as np
from keras.models import Sequential

from ..architecture import build_dense_regressor
from ..registry import register

# Relacion real que el modelo debe aprender. Las entradas y la nota estan
# normalizadas en [0, 1]: horas / 12 y nota / 100.
_TRUE_W = np.array([0.6, 0.3])
_TRUE_B = 0.1


def dataset() -> tuple[np.ndarray, np.ndarray]:
    """256 estudiantes sinteticos. RNG propio para no depender del orden."""
    rng = np.random.default_rng(2026)
    features = rng.uniform(0.0, 1.0, size=(256, 2))
    targets = features @ _TRUE_W + _TRUE_B
    return features, targets


@register(
    name="exam_score",
    input_features=2,
    feature_names=("hours_studied", "hours_slept"),
    dataset=dataset,
    epochs=300,
    seed=42,
    description="Nota normalizada (0-1) a partir de horas de estudio y sueno (/12)",
)
def build() -> Sequential:
    """Construye la arquitectura de 'exam_score'."""
    return build_dense_regressor(2)
```

What each part does:

- **`dataset()`** returns `(features, targets)` as NumPy arrays:
  `features` has shape `(samples, input_features)`, `targets` has shape
  `(samples,)`. Here the data is synthetic — `score = 0.6·studied +
  0.3·slept + 0.1` — so we know the right answer and can test for it. A real
  model would load a CSV or query Supabase here instead.
- **`np.random.default_rng(2026)`** is a *private* random generator. Don't
  use `np.random.uniform(...)` directly: the trainer resets NumPy's global
  seed before each model, so global randomness would make the data depend on
  which models are trained before this one.
- **`@register(...)`** declares the spec. It validates itself as soon as the
  file is imported: if `feature_names` had three names instead of two, every
  `import prue_model` would fail immediately with a clear message.
- **`build()`** returns a fresh, compiled, untrained model.
  `build_dense_regressor(2)` is a single linear `Dense` layer with 2 inputs
  and 1 output. Its input size must match `input_features`.

That is the only file you need. There is no list to edit:
`models/__init__.py` imports every module in its folder automatically.

> Tip: to park a half-finished model, name the file `_exam_score.py`. Files
> starting with `_` are skipped by the autoloader.

## 4. Check that it registered

```bash
python -c "import prue_model as p; print(p.list_names()); print(p.get_spec('exam_score'))"
```

You should see `('exam_score',)` followed by the spec. If you get a
`ValueError`, the spec is inconsistent — the message says which field.

## 5. Train it

```bash
python -m trainer --model exam_score
```

```
Modelo 'exam_score' guardado en apps\trainer\dist\exam_score.keras
```

TensorFlow prints some `oneDNN` / `cpu_feature_guard` log lines first; they
are harmless. The artifact lands in `MODELS_DIR` (by default
`apps/trainer/dist/`, which is git-ignored — artifacts are never committed).

`python -m trainer --all` retrains every registered model, including this one.

## 6. Serve it and predict

The API refuses to start unless **every** registered model has an artifact,
so train first (step 5). Then:

```bash
# Linux/macOS
export MODELS_DIR=apps/trainer/dist
# Windows (PowerShell)
$env:MODELS_DIR = "apps/trainer/dist"

uvicorn api.main:app --reload --app-dir apps/api/src
```

Interactive docs are at <http://127.0.0.1:8000/docs>. From another terminal:

**List the models**

```bash
curl http://127.0.0.1:8000/models
```

```json
[
  {
    "name": "exam_score",
    "input_features": 2,
    "feature_names": ["hours_studied", "hours_slept"],
    "description": "Nota normalizada (0-1) a partir de horas de estudio y sueno (/12)",
    "loaded": true,
    "task": "regression",
    "class_names": []
  }
]
```

**Predict** for a student who studied 8 hours and slept 8 hours
(`8 / 12 = 0.6667`):

```bash
curl -X POST http://127.0.0.1:8000/predict/exam_score \
     -H "content-type: application/json" \
     -d '{"features": [0.6667, 0.6667]}'
```

```json
{
  "model": "exam_score",
  "input": [0.6667, 0.6667],
  "prediction": 0.7052775025367737,
  "label": null,
  "probabilities": null
}
```

The true value is `0.6·0.6667 + 0.3·0.6667 + 0.1 = 0.70`, i.e. 70 points. The
model learned it. `label` and `probabilities` are `null` because this is a
regression.

> On Windows PowerShell, `curl` is an alias for `Invoke-WebRequest`. Use
> `curl.exe` with the same arguments, or:
> `Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict/exam_score -ContentType 'application/json' -Body '{"features":[0.6667,0.6667]}'`

**Bad input is rejected for free.** The input contract from the spec is
enforced without any API code:

```bash
curl -X POST http://127.0.0.1:8000/predict/exam_score \
     -H "content-type: application/json" -d '{"features": [0.5]}'
# 422 {"detail":"El modelo 'exam_score' espera 2 entradas (hours_studied, hours_slept); se recibieron 1."}

curl -X POST http://127.0.0.1:8000/predict/exam \
     -H "content-type: application/json" -d '{"features": [0.5]}'
# 404 {"detail":"El modelo 'exam' no existe. Disponibles: exam_score."}
```

## 7. Write its tests

Several existing tests already cover the new model automatically, because
they are parametrized over every registered spec (see
[models.md → Testing](models.md#testing)): shape checks, `--all` coverage,
reproducibility, and a full train-then-serve run.

What they can't know is what *this* model is supposed to do. Create
`packages/model/tests/test_models_exam_score.py`:

```python
"""Pruebas del modelo 'exam_score'."""

import numpy as np
import pytest
from prue_model import get_spec, load_model, predict
from trainer.train import train_spec

SPEC = get_spec("exam_score")


def test_spec_declares_the_input_contract():
    """Dos entradas, en este orden. Es lo que la API exige a los clientes."""
    assert SPEC.input_features == 2
    assert SPEC.feature_names == ("hours_studied", "hours_slept")
    assert SPEC.task == "regression"


def test_dataset_is_normalised_and_deterministic():
    """Entradas en [0, 1) y los mismos datos en cada llamada."""
    first, targets = SPEC.dataset()
    second, _ = SPEC.dataset()
    assert first.shape == (256, 2) and targets.shape == (256,)
    assert first.min() >= 0.0 and first.max() < 1.0
    np.testing.assert_array_equal(first, second)


@pytest.mark.slow
def test_trained_model_learns_the_relation(tmp_path):
    """8 h de estudio y 8 h de sueno -> 0.6*(8/12) + 0.3*(8/12) + 0.1 = 0.7."""
    model = load_model(train_spec(SPEC, tmp_path))
    result = predict(model, [8 / 12, 8 / 12], SPEC)
    assert result.value == pytest.approx(0.7, abs=0.05)
```

Notes:

- Training goes to pytest's `tmp_path`, never to `apps/trainer/dist`, so
  tests don't overwrite your real artifacts.
- The training test is marked `@pytest.mark.slow`. `pytest -m "not slow"`
  skips it for a fast feedback loop.
- Use a tolerance (`pytest.approx(..., abs=...)`) on learned values, never
  exact equality.

```bash
pytest packages/model/tests/test_models_exam_score.py -v
```

```
test_models_exam_score.py::test_spec_declares_the_input_contract PASSED
test_models_exam_score.py::test_dataset_is_normalised_and_deterministic PASSED
test_models_exam_score.py::test_trained_model_learns_the_relation PASSED
```

## 8. Run the whole suite

```bash
pytest -m "not slow"   # seconds: contracts, shapes, API validation
pytest                 # minutes: also trains every model
pylint packages/model/src apps/api/src apps/trainer/src
```

The full run trains your model a few more times (reproducibility,
train-then-serve). If it is slow to train, keep `epochs` as low as gives a
good result.

## 9. Commit

Commit exactly two files:

```
packages/model/src/prue_model/models/exam_score.py
packages/model/tests/test_models_exam_score.py
```

Not `apps/trainer/dist/exam_score.keras` — it is build output and is already
git-ignored. Anyone who pulls your change runs `python -m trainer --all`
before starting the API (the API tells them so if they forget).

## 10. Variant: a classifier

Now the same idea as a yes/no question: *will this student pass?* Three
things change compared to the regressor:

1. declare **`class_names`**: two names → binary, three or more → multiclass
2. targets in `dataset()` are **integer class indices** that line up with
   `class_names` (`0` → `"fail"`, `1` → `"pass"`)
3. build with **`build_dense_classifier(..., classes=N)`**, where `N` is
   `len(class_names)`

Create `packages/model/src/prue_model/models/exam_pass.py`:

```python
"""Modelo 'exam_pass': predice si un estudiante aprueba (clasificador binario)."""

import numpy as np
from keras.models import Sequential

from ..architecture import build_dense_classifier
from ..registry import register


def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Mismos estudiantes que 'exam_score'; aprueba quien saca 0.55 o mas."""
    rng = np.random.default_rng(2026)
    features = rng.uniform(0.0, 1.0, size=(256, 2))
    score = features @ np.array([0.6, 0.3]) + 0.1
    # Indices de clase, alineados con class_names: 0 = "fail", 1 = "pass".
    targets = (score >= 0.55).astype(int)
    return features, targets


@register(
    name="exam_pass",
    input_features=2,
    feature_names=("hours_studied", "hours_slept"),
    dataset=dataset,
    epochs=300,
    class_names=("fail", "pass"),
    description="Aprueba o reprueba a partir de horas de estudio y sueno (/12)",
)
def build() -> Sequential:
    """Construye la arquitectura de 'exam_pass'."""
    return build_dense_classifier(2, classes=2)
```

For a binary model the order of `class_names` matters: the sigmoid outputs
the probability of the **second** name, so put the "positive" class second.

Train it and restart the API (both models must have artifacts now):

```bash
python -m trainer --model exam_pass
uvicorn api.main:app --reload --app-dir apps/api/src
```

`GET /models` now reports `"task": "binary"` and
`"class_names": ["fail", "pass"]`, so a client knows how to read the answer
before asking. Predictions fill in the two classifier fields:

```bash
curl -X POST http://127.0.0.1:8000/predict/exam_pass \
     -H "content-type: application/json" -d '{"features": [0.6667, 0.6667]}'
```

```json
{
  "model": "exam_pass",
  "input": [0.6667, 0.6667],
  "prediction": 0.6725893020629883,
  "label": "pass",
  "probabilities": { "fail": 0.3274106979370117, "pass": 0.6725893020629883 }
}
```

A student who studied 2 hours (`0.1667`) and slept 6 (`0.5`):

```json
{
  "model": "exam_pass",
  "input": [0.1667, 0.5],
  "prediction": 0.3002713620662689,
  "label": "fail",
  "probabilities": { "fail": 0.6997286379337311, "pass": 0.3002713620662689 }
}
```

How to read the response by task:

| Field           | Regression | Binary | Multiclass |
|-----------------|------------|--------|------------|
| `prediction`    | the value  | probability of `class_names[1]` | index of the winning class |
| `label`         | `null`     | `class_names[1]` if `prediction >= 0.5`, else `class_names[0]` | winning class name |
| `probabilities` | `null`     | one entry per class | one entry per class |

For **multiclass**, e.g. `class_names=("low", "medium", "high")`, use targets
`0`, `1`, `2` and `build_dense_classifier(2, classes=3)`.

For tests, assert on `label` (and that the probabilities sum to 1) rather
than exact probability values.

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'prue_model'` | `PYTHONPATH` not set, or `:` used as separator on Windows. | Export it as in step 1. On Windows the separator is `;`. |
| Model missing from `list_names()` | File name starts with `_`, or it is outside `models/`. | Rename / move the file. |
| `ValueError: 'x' declara 2 entradas pero nombra 3` on import | `input_features` and `feature_names` disagree. | Make them match. |
| `DuplicateModelError` | Two files use the same `name` (or you reused a test-only name: `linear`, `plane`). | Pick a unique name. |
| Predictions are `NaN` or huge | `sgd` diverged on unscaled inputs. | Normalize to `[0, 1]`, or `build_dense_regressor(n, optimizer="adam")`. |
| API won't start: `Faltan artefactos en ...: exam_score` | Model registered but not trained. | `python -m trainer --all`. |
| API won't start: `MODELS_DIR no esta definida` | Env var missing. | Set `MODELS_DIR` (see [enviroment-variables.md](enviroment-variables.md)). |
| `OutputShapeError ... Reentrena el artefacto` (HTTP 500) | Changed `class_names` but kept the old `.keras`. | Retrain that model. |
| Every prediction returns 503 in a test | `TestClient(app)` used without `with`, so lifespan never ran. | Use the `client` fixture. |
| `test_all_matches_training_each_model_alone` fails for your model | `dataset()` uses global `np.random.*`. | Use `np.random.default_rng(seed)`. |
| Classifier always predicts the same class | Too few epochs, or unbalanced targets. | Raise `epochs`; check `np.bincount(targets)`. |

## 12. Checklist

- [ ] `packages/model/src/prue_model/models/<name>.py` with `dataset()`,
      `@register(...)` and `build()`
- [ ] `feature_names` in the order clients will send them; scaling noted in
      `description`
- [ ] `dataset()` uses a private RNG (`np.random.default_rng`)
- [ ] Classifier: `class_names` set, integer targets, matching `classes=`
- [ ] `python -m trainer --model <name>` succeeds
- [ ] `POST /predict/<name>` returns a sensible answer
- [ ] `packages/model/tests/test_models_<name>.py`, training tests marked
      `@pytest.mark.slow`
- [ ] `pytest` and `pylint` pass
- [ ] `.keras` artifact **not** committed
