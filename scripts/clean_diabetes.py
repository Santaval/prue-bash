"""Limpia y prepara el dataset M07 (riesgo de diabetes).

Uso desde la raíz del repositorio:
    python scripts/clean_diabetes.py

El archivo Excel original no se modifica. Se generan archivos en
``data/processed`` y un reporte en ``data/reports``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]
NUMERIC_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]
CATEGORICAL_COLUMNS = ["gender", "smoking_history"]


def parse_args() -> argparse.Namespace:
    """Lee las rutas configurables del comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
    default=Path("data/original/M07-M08-M09.xlsx"),
        help="Excel fuente que contiene la hoja Diabetes.",
    )
    parser.add_argument("--sheet", default="Diabetes")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_source(data: pd.DataFrame) -> None:
    """Falla temprano si el archivo no tiene el contrato esperado."""
    missing = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")


def clean(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Limpia el dataset y devuelve datos codificados junto con hallazgos."""
    original_rows = len(data)
    data = data[EXPECTED_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

    missing_before = data.isna().sum()
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)

    # "No Info" es una categoría válida del dataset: significa que el dato
    # sobre tabaquismo es desconocido, no que la fila deba eliminarse.
    data["smoking_history"] = data["smoking_history"].fillna("No Info")

    invalid_target = ~data["diabetes"].isin([0, 1])
    invalid_numeric = (
        data[["age", "bmi", "HbA1c_level", "blood_glucose_level"]] < 0
    ).any(axis=1)
    invalid_rows = invalid_target | invalid_numeric | data.isna().any(axis=1)
    invalid_count = int(invalid_rows.sum())
    data = data.loc[~invalid_rows].reset_index(drop=True)

    # Codificación estable para que entrenamiento e inferencia compartan el
    # mismo contrato: categorías conocidas quedan como enteros y desconocidas
    # pueden tratarse como -1 en una futura validación del API.
    for column in CATEGORICAL_COLUMNS:
        data[column] = pd.Categorical(data[column]).codes

    report = {
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicate_rows,
        "invalid_rows_removed": invalid_count,
        "final_rows": len(data),
        "missing_before_cleaning": {
            key: int(value) for key, value in missing_before.items() if value
        },
        "target_distribution": {
            str(key): int(value)
            for key, value in data["diabetes"].value_counts().sort_index().items()
        },
    }
    return data, report


def write_report(report: dict[str, object], path: Path) -> None:
    """Escribe un reporte legible y útil como evidencia de la etapa."""
    distribution = report["target_distribution"]
    lines = [
        "# Reporte de limpieza: M07 Riesgo de diabetes",
        "",
        "Fuente: `data/original/M07-M08-M09.xlsx`, hoja `Diabetes`.",
        "",
        "## Resultados",
        "",
        f"- Registros iniciales: {report['original_rows']}",
        f"- Duplicados eliminados: {report['duplicate_rows_removed']}",
        f"- Filas inválidas eliminadas: {report['invalid_rows_removed']}",
        f"- Registros finales: {report['final_rows']}",
        f"- Distribución de `diabetes`: {distribution}",
        "",
        "## Decisiones",
        "",
        "- `No Info` en `smoking_history` se conserva como categoría desconocida.",
        "- Las variables categóricas se codifican como enteros.",
        "- Se eliminan duplicados exactos.",
        "- Se eliminan filas con objetivo distinto de 0/1, valores numéricos negativos o campos faltantes restantes.",
        "- La separación de entrenamiento y prueba usa 80/20, semilla 42 y estratificación por `diabetes`.",
        "",
        "## Valores faltantes antes de limpiar",
        "",
    ]
    missing = report["missing_before_cleaning"]
    lines.extend(f"- `{key}`: {value}" for key, value in missing.items())
    if not missing:
        lines.append("- No se encontraron valores faltantes.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la limpieza completa."""
    args = parse_args()
    source = pd.read_excel(args.input, sheet_name=args.sheet)
    validate_source(source)
    data, report = clean(source)

    processed = Path("data/processed")
    reports = Path("data/reports")
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    data.to_csv(processed / "diabetes_clean.csv", index=False)
    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=args.seed,
        stratify=data["diabetes"],
    )
    train.to_csv(processed / "diabetes_train.csv", index=False)
    test.to_csv(processed / "diabetes_test.csv", index=False)
    write_report(report, reports / "diabetes_cleaning_report.md")

    print(f"Registros iniciales: {report['original_rows']}")
    print(f"Registros finales: {report['final_rows']}")
    print(f"Archivos generados en: {processed}")
    print(f"Reporte generado en: {reports / 'diabetes_cleaning_report.md'}")


if __name__ == "__main__":
    main()
