# Deploying with Docker Compose (Coolify)

`docker-compose.yaml` at the repository root runs the whole stack:

| Service | Image source             | Exposed          | Role |
|---------|--------------------------|------------------|------|
| `ui`    | `docker/ui/Dockerfile`   | public (port 80) | nginx serving `apps/ui`; proxies `/api/*` to the API |
| `api`   | `docker/api/Dockerfile`  | internal (8000)  | FastAPI + trained models; stores predictions in Supabase |

```
browser ──▶ ui (nginx :80) ──/api/*──▶ api (:8000) ──▶ Supabase (prediction_runs)
```

There is no database container: the API writes to the Supabase project
configured through `SUPABASE_URL` / `SUPABASE_KEY`. The `prediction_runs`
table must already exist there (`docs/sql/prediction_runs.sql`).

The browser only talks to `ui`. The UI calls the API under the same origin
(`https://<your-domain>/api`), so no CORS configuration is needed.

## How the models get into the image

The API image is built in two stages. The `trainer` stage runs
`scripts/clean_*.py` on `data/original/*.xlsx`, then
`python -m trainer --all --models-dir /models`. The final image copies only
`/models`, the API code and the model package. The first build is slow,
because it installs TensorFlow and trains the models; later builds reuse the
cache unless the data, code or requirements change.

## Coolify setup

1. **New Resource → Public/Private Repository**, pick this repo and branch.
2. **Build Pack: Docker Compose**. Base directory `/`, compose file
   `/docker-compose.yaml`.
3. Coolify detects the `ui` service through `SERVICE_FQDN_UI_80`. Set its
   domain (e.g. `https://prue.example.com`) in the `ui` service settings, or
   keep the generated one.
4. In **Environment Variables**, set `SUPABASE_URL` and `SUPABASE_KEY`
   (the server-side service role key). Both are required: the compose file
   marks them with `:?`, so the deploy fails fast if either is missing. Keep
   the key out of build-time variables; the API only needs it at runtime.
5. Deploy. `ui` comes up right away and answers `/api` with 502 until the
   API finishes loading the models.

### Environment variables

| Variable                 | Default                                  | Notes |
|--------------------------|------------------------------------------|-------|
| `SUPABASE_URL`           | —  (required)                            | URL of the Supabase project. |
| `SUPABASE_KEY`           | —  (required)                            | Service role key used for inserts. Never exposed to the browser. |
| `API_MODELS`             | `diabetes,academic_risk,loan_amount`     | Models the API loads. Each must have been trained in the image. |
| `TRAIN_ARGS`             | `--all`                                  | Build arg for the trainer, e.g. `--model diabetes` for a faster build. Keep it consistent with `API_MODELS`. |
| `API_SKIP_MODEL_LOADING` | `false`                                  | `true` starts the API without models (predictions return 503). |
| `API_CORS_ORIGINS`       | empty                                    | Only needed if another origin calls the API directly. |

Only `diabetes` predictions are stored in `prediction_runs`; the other
models answer without persisting.

## Running it elsewhere

Any machine with Docker works. The repo also has `compose.yaml` (local
Postgres only, for development), which `docker compose` picks by default, so
name the file explicitly:

```bash
SUPABASE_URL=... SUPABASE_KEY=... docker compose -f docker-compose.yaml up --build
# http://localhost is not published by default; add to the ui service:
#   ports: ["8080:80"]
```

## Notes

- `requirements.txt` includes dev tools (pytest, pylint); they end up in the
  API image. Splitting runtime and dev requirements would make it smaller.
