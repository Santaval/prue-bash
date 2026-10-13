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
| `SUPABASE_KEY` | `api`   | Supabase API key (anon or service role)             |
| `MODELS_DIR`   | `api`, `trainer` | Directory holding the trained `.keras` artifacts, one per model |

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

The API also refuses to start if *any* registered model's artifact is missing,
and it reports every missing name at once rather than one per restart. A
service that only fails on one route is worse than one that will not boot.

The trainer is more forgiving: when `MODELS_DIR` is unset it falls back to
`apps/trainer/dist` so a fresh clone can train without configuring anything.
The API never falls back. Note that the fallback is relative to the current
directory, so trainer commands are run from the repository root (or given an
explicit `--models-dir`).

Because it is a path and not a secret, `MODELS_DIR` is safe to commit in
`.example.env`.

## Notes

- Never share or expose the real keys in screenshots, commits, or documents.
- If someone else clones the repo, they should follow these same steps with their own credentials.