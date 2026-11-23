-- Ejecutar en Supabase SQL Editor o contra PostgreSQL local.
-- RLS queda habilitado sin políticas públicas. La conexión PostgreSQL directa
-- usa el propietario de la tabla; Supabase usa una clave de servidor.
create table if not exists public.prediction_runs (
    id uuid primary key,
    created_at timestamptz not null,
    model text not null check (model = 'diabetes'),
    input_data jsonb not null,
    prediction double precision not null check (prediction >= 0 and prediction <= 1),
    label text not null check (label in ('no_diabetes', 'diabetes')),
    probabilities jsonb not null
);

alter table public.prediction_runs enable row level security;
