# Environment Variables

This project uses environment variables for the Supabase credentials and for
locating the trained model artifacts. Never commit the real `.env` file to the
repository — only `.example.env` with empty values.

## Setup

1. Copy the template:
   ```bash
   cp .example.env .env
   ```

2. Fill in `.env` with your real values:
   ```
   SUPABASE_URL=your_url_here
   SUPABASE_KEY=your_key_here
   MODELS_DIR=apps/trainer/dist
   ```

3. Make sure `.env` is in `.gitignore`:
   ```
   .env
   ```

## Required variables

| Variable       | Used by | Description                                         |
|----------------|---------|-----------------------------------------------------|
| `SUPABASE_URL` | `api`   | URL of the Supabase project                         |
| `SUPABASE_KEY` | `api`   | Server-side service role key used for inserts; never expose it in the UI. |
| `MODELS_DIR`   | `api`, `trainer` | Directory holding the trained `.keras` artifacts, one per model |
| `API_SKIP_MODEL_LOADING` | `api` | Optional test switch. Set to `true` to start the API without loading model artifacts; valid prediction requests then return 503. Defaults to `false`. |
| `API_MODELS` | `api` | Optional comma-separated model names to load. Defaults to all registered models; use `diabetes` to serve M07 independently. |
| `PERSISTENCE_BACKEND` | `api` | `supabase` (default) or `postgres` for direct PostgreSQL connection. |
| `DATABASE_URL` | `api` | PostgreSQL connection URL; required only when `PERSISTENCE_BACKEND=postgres`. |
| `API_CORS_ORIGINS` | `api` | Optional comma-separated list of browser origins allowed by CORS (the UI origin). Defaults to `http://localhost:5173,http://127.0.0.1:5173`. |

## About `MODELS_DIR`

`MODELS_DIR` is a **directory**, not a file path. Every registered model
resolves to `<MODELS_DIR>/<name>.keras`, so adding a model requires no change
here at all — that is the whole reason it is a directory. Registering a
model named `exam_score` is enough for the API to start looking for
`apps/trainer/dist/exam_score.keras`.

The API refuses to start if `MODELS_DIR` is unset, and the trainer must have run
at least once so the files exist. This is deliberate: the path used to be
hard-coded as a relative walk up the directory tree, which broke silently
whenever a file moved. Passing it as configuration means the dependency between
the API and the training output is explicit — and once Turborepo is wired up, it
can guarantee the directory is populated before the API starts.

The API refuses to start if an artifact selected by `API_MODELS` is missing,
and reports all missing selected artifacts together. When the variable is unset,
all registered models are selected.

The trainer is more forgiving: when `MODELS_DIR` is unset it falls back to
`apps/trainer/dist` so a fresh clone can train without configuring anything.
The API never falls back. Note that the fallback is relative to the current
directory, so trainer commands are run from the repository root (or given an
explicit `--models-dir`).

Because it is a path and not a secret, `MODELS_DIR` is safe to commit in
`.example.env`.

## Starting the API without model artifacts

For UI or request-contract development, set `API_SKIP_MODEL_LOADING=true` before
starting the API. The server starts without `MODELS_DIR` artifacts, so routes
and request validation can be exercised. Valid prediction requests return 503
because no model is loaded; invalid request bodies still return 422. Leave this
flag unset or set it to `false` for normal inference.

## Persisting diabetes predictions

Run [`sql/prediction_runs.sql`](sql/prediction_runs.sql) in the Supabase SQL
Editor to create the table and enable row-level security. Configure
`SUPABASE_URL` and the server-side `SUPABASE_KEY` in the API environment. The
M07 endpoint returns 503 if it cannot persist a prediction. The key belongs
only on the API server, never in browser code.

For local PostgreSQL, use `PERSISTENCE_BACKEND=postgres` and set
`DATABASE_URL`. The Supabase backend remains the default and is selected when
the flag is unset or set to `supabase`.

## Notes

- Never share or expose the real keys in screenshots, commits, or documents.
- If someone else clones the repo, they should follow these same steps with their own credentials.
