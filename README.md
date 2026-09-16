# Prue-bash

## Description

This is a solution integrating UI, API, and a Machine Learning model using PostgreSQL through Supabase. This will be used to evaluate the testing practices of the team.

## Team Members

- Ignacio Cerdas — C21988
- Aarón Santana Valdelomar — C27373
- Katherine Acosta — B70047
- Aldo Cascante — C21988

## Project Objective


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
   cp .env.example .env
   ```
   Fill out `.env` with your Supabase project credentials (do not share or upload this file to the repository).

## Running the Project

Run tests with:
```bash
pytest
```

Run static code analysis with:
```bash
pylint src/
```

## Repository Structure

```
prue-bash/
│
├── README.md          # This file
├── requirements.txt   # Project dependencies
├── .env.example       # Environment variables template (without real values)
│
├── src/               # Project source code
├── tests/             # Automated tests (pytest)
├── docs/              # Documentation (Test Plan, Test Design, etc.)
└── evidencias/        # Setup and environment verification evidence
```