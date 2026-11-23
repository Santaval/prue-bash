# Documentation

| Document | Read it when |
|----------|--------------|
| [adding-a-model.md](adding-a-model.md) | You want to add a model. Step-by-step tutorial: file → train → serve → test → commit, with a regression and a classifier example. |
| [models.md](models.md) | You need to know how it works: `ModelSpec`, registry and autoload, architecture builders, input/output contracts, trainer, API, reproducibility, errors, test coverage, and the M07 request example. |
| [enviroment-variables.md](enviroment-variables.md) | You are setting up `.env`, or the API complains about `MODELS_DIR`. |
| [postgres-local.md](postgres-local.md) | You want to run the persistence layer against local PostgreSQL in Podman. |
| [evidence/diabetes-model-metrics.md](evidence/diabetes-model-metrics.md) | You need the current holdout metrics and confusion matrix for M07. |

The model registry auto-loads model modules from
`packages/model/src/prue_model/models/`. Current entries include
`academic_risk`, `diabetes`, and `loan_amount`.
