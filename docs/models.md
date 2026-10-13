# How the model system works

This document explains the moving parts behind every model in the repository:
how a model is declared, how it gets trained, how the API serves it, and how
the test suite covers it. If you just want to ship a model, start with
[adding-a-model.md](adding-a-model.md) and come back here when you need the
details.

> **Current state:** the repository ships with **no production models**. The
> folder `packages/model/src/prue_model/models/` contains only its
> `__init__.py`. The test suite registers two throwaway models of its own
> (see [Testing](#testing)), so everything is exercised even while the
> registry is empty.

## Contents

- [The big picture](#the-big-picture)
- [The `ModelSpec`: one declaration per model](#the-modelspec-one-declaration-per-model)
- [Registration and autoload](#registration-and-autoload)
- [Architecture builders](#architecture-builders)
- [The input contract](#the-input-contract)
- [The output contract](#the-output-contract)
- [Artifacts and `MODELS_DIR`](#artifacts-and-models_dir)
- [The trainer](#the-trainer)
- [The API](#the-api)
- [Reproducibility](#reproducibility)
- [Errors](#errors)
- [Testing](#testing)

## The big picture

```
                  packages/model  (prue_model)
   ┌──────────────────────────────────────────────────────────┐
   │  models/<name>.py ──@register──▶ registry (ModelSpec)    │
   │                                       │                  │
   │  architecture.py   inference.py   artifacts.py           │
   └───────────┬───────────────────────────┬──────────────────┘
               │                           │
     apps/trainer                     apps/api
   python -m trainer               uvicorn api.main:app
   reads the registry,             reads the registry, loads
   trains each spec, writes  ───▶  <MODELS_DIR>/<name>.keras,
   <MODELS_DIR>/<name>.keras       serves POST /predict/<name>
```

The two applications never talk to each other directly. They agree on
everything through the shared library:

| Question                          | Answered by                         |
|-----------------------------------|-------------------------------------|
| Which models exist?               | `registry.list_specs()`             |
| What does model X take as input?  | `ModelSpec.input_features` / `feature_names` |
| What does model X return?         | `ModelSpec.class_names` → `task`, `output_units` |
| Where is model X's trained file?  | `artifacts.artifact_path(spec)`     |
| Is this input vector valid?       | `inference.to_batch()`              |
| How do I read the raw output?     | `inference.interpret()`             |

Because each answer lives in exactly one place, the trainer and the API cannot
drift apart.

## The `ModelSpec`: one declaration per model

`ModelSpec` (`packages/model/src/prue_model/registry.py`) is a frozen dataclass
that holds everything a model says about itself:

| Field            | Type                       | Required | Meaning |
|------------------|----------------------------|----------|---------|
| `name`           | `str`                      | yes      | Unique id. Used in the URL (`/predict/<name>`), the CLI (`--model <name>`) and the file name (`<name>.keras`). Case-sensitive. |
| `input_features` | `int`                      | yes      | Number of input values. Must be `>= 1`. |
| `feature_names`  | `tuple[str, ...]`          | yes      | One name per input, **in order**. Length must equal `input_features`. Shown in `GET /models` and in 422 error messages. |
| `build`          | `() -> keras.Model`        | yes      | Builds and compiles a fresh, untrained model. This is the function you decorate. |
| `dataset`        | `() -> (features, targets)`| yes      | Returns the training data as two NumPy arrays. |
| `epochs`         | `int`                      | yes      | Training epochs. Must be `>= 1`. |
| `seed`           | `int`                      | no (42)  | Random seed applied right before `build()` is called. |
| `description`    | `str`                      | no ("")  | Free text shown in `GET /models`. |
| `class_names`    | `tuple[str, ...]`          | no (())  | Empty for regression; 2+ names for classification. |

Two properties are **derived**, never declared, so they cannot contradict
`class_names`:

| `class_names`      | `task`        | `output_units` |
|--------------------|---------------|----------------|
| `()`               | `regression`  | `1`            |
| `("a", "b")`       | `binary`      | `1`            |
| `("a", "b", "c")`  | `multiclass`  | `3` (one per class) |

The spec validates itself in `__post_init__`, which runs **when the model file
is imported**. A malformed spec therefore fails at startup (of the trainer, the
API, or pytest), not on the first request. It rejects:

- `input_features < 1`
- `len(feature_names) != input_features`
- `epochs < 1`
- exactly one class name (neither regression nor classification)
- duplicated class names (they become dictionary keys in the response)

## Registration and autoload

You never construct a `ModelSpec` by hand. You decorate the model's `build`
function with `@register(...)`, passing every field except `build`:

```python
@register(name="exam_score", input_features=2,
          feature_names=("hours_studied", "hours_slept"),
          dataset=dataset, epochs=300)
def build() -> Sequential:
    return build_dense_regressor(2)
```

`register` stores the spec in a module-level dictionary and returns `build`
**unchanged**, so `build` stays importable and testable on its own. Registering
the same name twice raises `DuplicateModelError`.

The registry is filled automatically. `prue_model/__init__.py` imports the
`models` package, and `models/__init__.py` imports every sibling module:

```python
for module in pkgutil.iter_modules(__path__):
    if module.name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{module.name}")
```

Consequences:

- **Adding a model is adding one file.** There is no list to update.
- **Files starting with `_` are skipped.** Rename `exam_score.py` to
  `_exam_score.py` to park a half-written model without deleting it.
- **Removing a model is deleting its file** (plus its tests and, optionally,
  its `.keras` artifact).
- Model files must not have import-time side effects beyond registering (no
  training, no I/O). `import prue_model` has to stay cheap and safe.

Query functions:

| Function        | Returns |
|-----------------|---------|
| `get_spec(name)`| The spec, or raises `ModelNotFoundError` listing available names. |
| `list_specs()`  | All specs as a tuple, sorted by name. |
| `list_names()`  | All names as a tuple, sorted. |

Sorting makes the output stable: `--all` always trains in the same order and
`GET /models` always lists in the same order.

## Architecture builders

`architecture.py` offers two side-effect-free builders. Each returns a
compiled but untrained `Sequential` model. You are not forced to use them —
`build` may return any compiled `keras.Model` — but the output layer must match
`spec.output_units` (a parametrized test checks this for every model).

### `build_dense_regressor(input_features, *, units=1, optimizer="sgd", loss="mean_squared_error")`

A single `Dense` layer with linear activation: plain linear regression.

- `sgd` at its default learning rate **diverges to `NaN`** on real-scale data.
  Normalize the inputs to roughly `[0, 1]`, or pass `optimizer="adam"`.

### `build_dense_classifier(input_features, *, classes, optimizer="adam")`

A single `Dense` layer whose head depends on `classes`:

| `classes` | Units | Activation | Loss |
|-----------|-------|------------|------|
| `2`       | 1     | `sigmoid`  | `binary_crossentropy` |
| `>= 3`    | `classes` | `softmax` | `sparse_categorical_crossentropy` |

This mirrors `ModelSpec.output_units` exactly. Pass the same number of classes
as you declare in `class_names`.

### Writing a custom architecture

For anything deeper, write the layers yourself inside `build`:

```python
def build() -> Sequential:
    model = Sequential([
        keras.Input(shape=(4,)),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax"),   # == len(class_names)
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model
```

Always start with `keras.Input(shape=(input_features,))` so the input shape is
known before training.

## The input contract

Clients send a flat list of numbers, in the order of `feature_names`:

```json
{ "features": [0.66, 0.58] }
```

`inference.to_batch(features, spec)` is the **only** place that validates it.
It rejects, with `InvalidFeaturesError` (→ HTTP 422):

- anything that is not a list/sequence, including strings (`"abc"` is a
  sequence of length 3 and would otherwise sneak through)
- the wrong number of values — the message names the expected features:
  `El modelo 'exam_score' espera 2 entradas (hours_studied, hours_slept); se recibieron 1.`
- non-numeric values (`null`, `"x"`, nested lists)
- `NaN`, `Infinity`, `-Infinity` (Pydantic accepts them; a `NaN` prediction
  is not even valid JSON)

On success it returns a `float32` array of shape `(1, input_features)`.
Integers are accepted and coerced.

The library does **not** normalize anything. If your model was trained on
`hours / 12`, the client must send `hours / 12`. Document the scaling in the
spec's `description` and `feature_names`.

## The output contract

`inference.interpret(raw, spec)` turns the raw model output into a
`Prediction(value, label, probabilities)`:

| Task         | `value`                                  | `label`                         | `probabilities` |
|--------------|------------------------------------------|---------------------------------|-----------------|
| `regression` | the predicted number                     | `null`                          | `null`          |
| `binary`     | probability of the **second** class name | second name if `value >= 0.5`, else first | `{first: 1 - p, second: p}` |
| `multiclass` | index of the winning class (as float)    | `class_names[index]`            | one entry per class |

All numbers are native Python floats, so they serialize to JSON. If the
artifact's output width does not match the spec (e.g. you added a class but
did not retrain), `interpret` raises `OutputShapeError` telling you to retrain,
instead of failing with an `IndexError`.

**Ordering matters for classifiers.** Training targets are integer class
indices that line up positionally with `class_names`: target `0` is
`class_names[0]`, target `1` is `class_names[1]`, and so on. For a binary
model the sigmoid outputs the probability of `class_names[1]`, so put the
"positive" class second.

## Artifacts and `MODELS_DIR`

Each trained model is saved as `<MODELS_DIR>/<name>.keras`, resolved by
`artifacts.artifact_path(spec, base=None)`. `MODELS_DIR` is a directory, so
adding a model never requires touching configuration. See
[enviroment-variables.md](enviroment-variables.md) for the full rules; in
short:

- The **trainer** falls back to `apps/trainer/dist` if `MODELS_DIR` is unset.
- The **API** refuses to start if `MODELS_DIR` is unset.
- `apps/trainer/dist/` is git-ignored: artifacts are build output, never
  committed.

## The trainer

`apps/trainer` is a thin CLI over the registry:

```bash
python -m trainer --model <name>           # one model
python -m trainer --all                    # every registered model
python -m trainer --all --models-dir PATH  # override MODELS_DIR
```

`--model` and `--all` are mutually exclusive and one is required. An unknown
name exits with code `2` and lists the available names.

For each spec, `train_spec(spec, base_dir)`:

1. calls `keras.utils.set_random_seed(spec.seed)`
2. calls `spec.build()` to get a fresh compiled model
3. calls `spec.dataset()` to get `(features, targets)`
4. runs `model.fit(features, targets, epochs=spec.epochs, verbose=0)`
5. saves to `artifact_path(spec, base_dir)`, creating the folder if needed

`fit` uses Keras defaults for everything else (batch size 32, no validation
split, no callbacks). If a model needs more control, keep the extra logic in
its own module and expose it through `build`/`dataset`; do not special-case
models inside the trainer.

## The API

`apps/api` (FastAPI). On startup, `lifespan` calls `deps.load_all_models()`,
which:

1. requires `MODELS_DIR`
2. checks that **every** registered model has an artifact, and reports all
   missing names in a single error
3. loads them all into memory

It is all-or-nothing on purpose: a service where one route silently answers
503 is worse than one that refuses to boot.

| Route                    | Description |
|--------------------------|-------------|
| `GET /`                  | Health check. |
| `GET /models`            | Every registered model: `name`, `input_features`, `feature_names`, `description`, `loaded`, `task`, `class_names`. |
| `POST /predict/{name}`   | Body `{"features": [...]}`. Returns `model`, `input`, `prediction`, `label`, `probabilities`. |

Status codes for `POST /predict/{name}`:

| Code | When |
|------|------|
| 200  | Prediction returned. |
| 404  | `name` is not registered (checked first; the detail lists available names). |
| 422  | Body is malformed or violates the input contract. |
| 503  | The model is registered but not loaded (only happens if lifespan did not run). |

## Reproducibility

The seed is re-applied **per model, right before `build()`**, because weight
initialization consumes the RNG too. Therefore:

- training the same model twice gives bit-identical weights
- `--all` gives exactly the same weights as training each model alone,
  regardless of order

Two rules keep this true for your model:

- **Use a private RNG in `dataset()`** — `np.random.default_rng(<seed>)`. Never
  use module-level `np.random.*`: the trainer resets the global seed between
  models, so your data would depend on the training order.
- **Never compare artifacts by hash.** A `.keras` file is a zip archive with
  variable metadata; identical weights still produce different bytes. Load
  the model and compare `get_weights()`.

## Errors

All in `prue_model.errors`, all subclasses of `PrueModelError`:

| Exception               | Raised when | API mapping |
|-------------------------|-------------|-------------|
| `ModelNotFoundError`    | `get_spec` gets an unknown name | 404 |
| `DuplicateModelError`   | two files register the same name | startup fails |
| `InvalidFeaturesError`  | `to_batch` rejects the input (also a `ValueError`) | 422 |
| `ArtifactNotFoundError` | `load_model` cannot find the `.keras` file (also a `FileNotFoundError`) | startup fails |
| `OutputShapeError`      | artifact output width ≠ `spec.output_units` | 500 — retrain |

## Testing

### What you get for free

Several tests are parametrized over `list_specs()`, so **a new model is
covered the moment its file exists**:

| Test | Checks |
|------|--------|
| `packages/model/tests/test_registry.py::test_every_spec_builds_its_declared_shape` | `build()` input/output shapes match `input_features` / `output_units`. |
| `packages/model/tests/test_registry.py::test_every_dataset_matches_its_declared_shape` | `dataset()` has `input_features` columns. |
| `tests/e2e/test_reproducibility.py::test_all_matches_training_each_model_alone` | `--all` and solo training produce identical weights. |
| `tests/e2e/test_train_then_serve.py::test_train_then_serve` | Train everything, start the API, `POST /predict/<name>` returns a finite number. |
| `apps/api/tests/test_deps.py` | The API refuses to start if your artifact is missing. |
| `apps/trainer/tests/test_cli.py::test_all_covers_every_registered_model` | `--all` includes your model. |

### What you should add

A `packages/model/tests/test_models_<name>.py` that pins down the behaviour
specific to your model: the declared contract, the dataset's shape and range,
and that the trained model actually learns what it should. Mark tests that
train with `@pytest.mark.slow` so `pytest -m "not slow"` stays fast. See the
tutorial for a full example.

### The test-only models

The repository has no production models, but the API and e2e suites need a
populated registry. The root `conftest.py` registers two models at import
time, before collection:

| Name     | Inputs        | Relation |
|----------|---------------|----------|
| `linear` | `x`           | `y = 2x - 1` |
| `plane`  | `a`, `b`, `c` | `y = 2a - 1.5b + 0.5c + 0.25` |

They exist **only inside pytest**: the real trainer and API never see them.
Their targets are noise-free, so `tests/e2e/test_training_converges.py` can
assert the exact coefficients. Don't reuse these names for real models — the
registry would raise `DuplicateModelError` during tests.

### Fixtures worth knowing

| Fixture              | Scope   | Purpose |
|----------------------|---------|---------|
| `trained_models_dir` | session | Trains every registered model once into a temp dir. |
| `client`             | test    | `TestClient` with `MODELS_DIR` pointing at `trained_models_dir` and lifespan executed. |
| `registry_snapshot`  | test    | Restores the registry afterwards, so specs registered inside a test do not leak. |
| `fake_model`, `fake_binary_model`, `fake_multiclass_model` | test | Objects with a `.predict()` for testing inference without Keras. |

`TestClient(app)` used **outside** a `with` block does not run lifespan, so
nothing is loaded and every prediction returns 503. Always use the `client`
fixture (or `with TestClient(app) as client:`).
