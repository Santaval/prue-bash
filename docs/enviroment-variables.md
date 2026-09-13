# Environment Variables

This project uses environment variables for the Supabase credentials. Never commit the real `.env` file to the repository — only `.env.example` with empty values.

## Setup

1. Copy the template:
   ```bash
   cp .env.example .env
   ```

2. Fill in `.env` with your real values:
   ```
   SUPABASE_URL=your_url_here
   SUPABASE_KEY=your_key_here
   ```

3. Make sure `.env` is in `.gitignore`:
   ```
   .env
   ```

## Required variables

| Variable       | Description                                  |
|----------------|-----------------------------------------------|
| `SUPABASE_URL` | URL of the Supabase project                   |
| `SUPABASE_KEY` | Supabase API key (anon or service role)       |

## Notes

- Never share or expose the real keys in screenshots, commits, or documents.
- If someone else clones the repo, they should follow these same steps with their own credentials.