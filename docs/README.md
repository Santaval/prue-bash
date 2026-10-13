# Documentation

| Document | Read it when |
|----------|--------------|
| [adding-a-model.md](adding-a-model.md) | You want to add a model. Step-by-step tutorial: file → train → serve → test → commit, with a regression and a classifier example. |
| [models.md](models.md) | You need to know how it works: `ModelSpec`, registry and autoload, architecture builders, input/output contracts, trainer, API, reproducibility, errors and test coverage. |
| [enviroment-variables.md](enviroment-variables.md) | You are setting up `.env`, or the API complains about `MODELS_DIR`. |

The repository currently ships with **no production models**: the registry is
empty until someone adds a file under
`packages/model/src/prue_model/models/`.
