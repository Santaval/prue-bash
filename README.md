# Prue-bash

## Description

This is a solution integrating UI, API, and a Machine Learning model using PostgreSQL through Supabase. This will be used to evaluate the testing practices of the team.

The repository is organized as a monorepo: two runnable applications (`api`, `trainer`) plus a UI, sharing a single library (`packages/model`) that owns the model architecture and the inference contract.

## Team Members

- Ignacio Cerdas — C21988
- Aarón Santana Valdelomar — C27373
- Katherine Acosta — B70047
- Aldo Cascante — C21988

## Project Objectives

- Develop a software project that includes a robust testing plan closely aligned with the business domain.
- Create tests to validate the system's most granular aspects.
- Create tests to validate workflows that reflect business logic.
- Establish a clear plan to ensure the optimal division and utilization of available resources.
- Design test cases that account for anomalous inputs to ensure that workflows do not produce unexpected results when processing them.

## Environment Setup

### Prerequisites

- Python 3.13 or higher
- pip
- Supabase account/project

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Santaval/prue-bash.git
   cd prue-bash
   ```

2. Create and activate the virtual environment:
   ```bash
   python -m venv .venv

   # Linux/macOS
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .example.env .env
   ```
   Fill out `.env` with your Supabase project credentials (do not share or upload this file to the repository). See [docs/enviroment-variables.md](docs/enviroment-variables.md) for the full list.

5. Make the local packages importable. Until the workspace configuration lands
   (see *Pending work* below), this is done with `PYTHONPATH`:
   ```bash
   # Linux/macOS
   export PYTHONPATH="packages/model/src:apps/api/src:apps/trainer/src"

   # Windows (PowerShell)
   $env:PYTHONPATH = "packages/model/src;apps/api/src;apps/trainer/src"
   ```
   On Windows the separator is `;` even inside Git Bash — using `:` there fails
   with `ModuleNotFoundError: No module named 'prue_model'`.

   Running `pytest` needs none of this: `pytest.ini` declares the same three
   source roots.

## Running the Project

All commands are run from the repository root.

The repository ships with **no models** — see
[docs/adding-a-model.md](docs/adding-a-model.md) to add one.

Train the models (writes one `.keras` artifact per model into `MODELS_DIR`,
by default `apps/trainer/dist/`):
```bash
python -m trainer --all
python -m trainer --model <name>           # just one
python -m trainer --all --models-dir /tmp  # somewhere else
```

Serve the API (requires every model's artifact and `MODELS_DIR`):
```bash
uvicorn api.main:app --reload --app-dir apps/api/src
```

Verify the Supabase connection:
```bash
python apps/api/src/api/db.py
```

Run tests:
```bash
pytest                # everything
pytest -m "not slow"  # skips the tests that train a model
```

Run static code analysis:
```bash
pylint packages/model/src apps/api/src apps/trainer/src
```

## Repository Structure

```
prue-bash/
│
├── README.md              # This file
├── requirements.txt       # Project dependencies
├── .example.env           # Environment variables template (without real values)
│
├── pytest.ini             # Test config: source roots, markers
├── conftest.py            # Shared fixtures
│
├── apps/
│   ├── api/               # FastAPI service: GET /models, POST /predict/{name}
│   │   ├── src/api/
│   │   │   ├── main.py    # Application wiring and routes
│   │   │   ├── schemas.py # Request/response models
│   │   │   ├── deps.py    # Loads every registered model from MODELS_DIR
│   │   │   └── db.py      # Supabase client
│   │   └── tests/         # Unit and integration tests for the API
│   │
│   ├── trainer/           # Training job, driven by the registry
│   │   ├── src/trainer/
│   │   │   ├── __main__.py  # python -m trainer
│   │   │   ├── cli.py       # --model NAME | --all | --models-dir PATH
│   │   │   └── train.py     # train_spec(), train_all()
│   │   ├── dist/          # Build output: one .keras per model (git-ignored)
│   │   └── tests/
│   │
│   └── ui/                # Vite + React front-end (not started yet)
│
├── packages/
│   └── model/             # Shared library: registry + architecture + inference
│       ├── src/prue_model/
│       │   ├── registry.py      # ModelSpec, register(), get_spec()
│       │   ├── architecture.py  # build_dense_regressor/classifier()
│       │   ├── inference.py     # load_model(), to_batch(), predict()
│       │   ├── artifacts.py     # MODELS_DIR resolution
│       │   ├── errors.py        # Domain exceptions
│       │   └── models/          # One file per model (empty: add yours here)
│       └── tests/
│
├── tests/e2e/             # Cross-application workflow tests
├── docs/                  # Model system, adding a model, environment variables
└── evidence/              # Setup and environment verification evidence
```

### Why the model lives in `packages/`

Both the trainer and the API need to agree on each model's input shape. That
contract is declared exactly once per model, in its `ModelSpec`
(`input_features` and `feature_names`), and `to_batch()` in `inference.py` is
the single place that enforces it — so the trainer and the request handler
cannot drift apart by construction rather than by convention. The same library
also owns artifact-path resolution, so the two sides cannot disagree about
where a `.keras` file lives either.

The **output** contract lives in the same spec. A model declares `class_names`,
and its task follows from that: empty means regression, two names mean binary
classification, three or more mean multiclass. `ModelSpec.task` and
`ModelSpec.output_units` are derived rather than declared, so the head the
trainer builds and the way the API reads a prediction cannot contradict each
other — `interpret()` in `inference.py` is the one place that turns a raw
output tensor into a value, a label and a probability per class.

Specs are plain declarations with no side effects, so the architecture stays
importable without triggering a training run, which is what allows it to be
unit tested.

`apps/trainer/dist/*.keras` are build artifacts, not source. They are
git-ignored and regenerated by the training job.

### Adding a model

Adding a model is adding **one file** under
`packages/model/src/prue_model/models/`. Nothing else in the repository
changes — not the trainer, not the API, not the environment.

- [docs/adding-a-model.md](docs/adding-a-model.md) — end-to-end tutorial
  (create, train, serve, test, commit), with a regression and a classifier.
- [docs/models.md](docs/models.md) — how the registry, contracts, trainer and
  API fit together.

### Reproducibility of the training job

The trainer sets a fixed seed, so **the trained weights are identical across
runs** — verified as bit-identical by `tests/e2e/test_reproducibility.py`. This matters
because caching a training step is only safe if the step is deterministic.

The seed is declared per model and re-applied immediately before that model is
built, because weight initialization consumes the RNG too. That is what makes
`python -m trainer --all` produce exactly the same weights as training each
model on its own: the result does not depend on the training order.

The `.keras` **file bytes are not stable**, however: the format is a zip archive
carrying variable metadata, so two runs with identical weights still produce
different file hashes. Practical consequence: never assert reproducibility by
hashing the artifact — load it and compare `get_weights()` instead.

## Pending work

- Per-package `pyproject.toml` files and a `uv` workspace, so `apps/api` depends
  on `prue_model` explicitly and `PYTHONPATH` is no longer needed.
- Turborepo task orchestration (`turbo.json`), to declare that the API depends
  on the training output and to cache training runs.
- The `apps/ui` application.