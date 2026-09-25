"""Interfaz de linea de comandos del entrenador."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from prue_model import ModelNotFoundError, get_spec, list_specs, models_dir

from .train import train_spec


def build_parser() -> argparse.ArgumentParser:
    """Arma el parser: hay que elegir --model o --all, nunca ambos."""
    parser = argparse.ArgumentParser(
        prog="python -m trainer",
        description="Entrena los modelos declarados en el registro.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", metavar="NAME", help="Nombre del modelo a entrenar.")
    target.add_argument(
        "--all", action="store_true", help="Entrena todos los modelos registrados."
    )
    parser.add_argument(
        "--models-dir",
        metavar="PATH",
        default=None,
        help=(
            "Directorio donde escribir los artefactos. Por defecto usa "
            "MODELS_DIR, y si no esta definida, apps/trainer/dist relativo al "
            "directorio actual: corre este comando desde la raiz del repo."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada. Devuelve el codigo de salida del proceso."""
    args = build_parser().parse_args(argv)
    base = Path(args.models_dir) if args.models_dir else models_dir(required=False)

    try:
        specs = list_specs() if args.all else (get_spec(args.model),)
    except ModelNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for spec in specs:
        destination = train_spec(spec, base)
        print(f"Modelo '{spec.name}' guardado en {destination}")
    return 0
