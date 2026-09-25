# Environment Variables

This project uses environment variables for the Supabase credentials and for
locating the trained model artifact. Never commit the real `.env` file to the
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
   MODEL_PATH=apps/trainer/dist/math.keras
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
| `MODEL_PATH`   | `api`   | Path to the trained `.keras` artifact               |

## About `MODEL_PATH`

The API refuses to start if `MODEL_PATH` is unset, and the trainer must have run
at least once so the file exists. This is deliberate: the path used to be
hard-coded as a relative walk up the directory tree, which broke silently
whenever a file moved. Passing it as configuration means the dependency between
the API and the training output is explicit — and once Turborepo is wired up, it
can guarantee the artifact exists before the API starts.

Because it is a path and not a secret, `MODEL_PATH` is safe to commit in
`.example.env`.

## Notes

- Never share or expose the real keys in screenshots, commits, or documents.
- If someone else clones the repo, they should follow these same steps with their own credentials.