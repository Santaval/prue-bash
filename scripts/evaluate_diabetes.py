"""Evalúa el artefacto M07 en el split de prueba y genera evidencia Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from prue_model import artifact_path, get_spec, load_model, models_dir


def parse_args() -> argparse.Namespace:
    """Define rutas y umbral para una evaluación reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/diabetes_test.csv"),
        help="CSV de prueba generado por clean_diabetes.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/diabetes-model-metrics.md"),
        help="Reporte Markdown que se generará.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def evaluate(test_path: Path, threshold: float) -> dict[str, object]:
    """Carga el modelo y calcula métricas para la clase positiva diabetes."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold debe estar entre 0 y 1.")

    data = np.loadtxt(test_path, delimiter=",", skiprows=1, dtype=np.float32)
    features, targets = data[:, :-1], data[:, -1].astype(int)
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("El conjunto de prueba debe contener las clases 0 y 1.")

    spec = get_spec("diabetes")
    base_dir = models_dir(required=False)
    model_path = artifact_path(spec, base_dir)
    model = load_model(model_path)
    probabilities = model.predict(features, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()

    return {
        "model_path": model_path,
        "test_path": test_path,
        "samples": len(targets),
        "actual_negative": int((targets == 0).sum()),
        "actual_positive": int((targets == 1).sum()),
        "predicted_negative": int((predictions == 0).sum()),
        "predicted_positive": int((predictions == 1).sum()),
        "threshold": threshold,
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "roc_auc": roc_auc_score(targets, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_report(metrics: dict[str, object], output: Path) -> None:
    """Escribe las métricas junto con el alcance y las limitaciones."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Métricas del modelo M07: riesgo de diabetes",
        "",
        f"- Artefacto: `{metrics['model_path']}`",
        f"- Conjunto de prueba: `{metrics['test_path']}`",
        f"- Muestras: {metrics['samples']}",
        f"- Distribución real: clase 0 = {metrics['actual_negative']}; "
        f"clase 1 = {metrics['actual_positive']}",
        f"- Umbral de clasificación: {metrics['threshold']}",
        "",
        "## Resultados",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision (diabetes) | {metrics['precision']:.4f} |",
        f"| Recall (diabetes) | {metrics['recall']:.4f} |",
        f"| F1 (diabetes) | {metrics['f1']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        "",
        "## Matriz de confusión",
        "",
        "Filas = clase real; columnas = clase predicha.",
        "",
        "| Real \\ Predicha | No diabetes (0) | Diabetes (1) |",
        "|---|---:|---:|",
        f"| No diabetes (0) | {metrics['tn']} | {metrics['fp']} |",
        f"| Diabetes (1) | {metrics['fn']} | {metrics['tp']} |",
        "",
        "## Interpretación y límites",
        "",
        "El conjunto está desbalanceado, por lo que accuracy por sí sola no "
        "describe el desempeño. Con el umbral configurado, recall mide la "
        "fracción de positivos detectados y precision la fracción de "
        "predicciones positivas que fueron correctas.",
        "",
        "Estas métricas corresponden a un único split de prueba del dataset "
        "disponible. No constituyen validación clínica ni garantizan el mismo "
        "desempeño en otras poblaciones; la predicción no es un diagnóstico.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Evalúa M07 y genera el reporte de métricas."""
    args = parse_args()
    metrics = evaluate(args.test, args.threshold)
    write_report(metrics, args.output)
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{name}: {metrics[name]:.4f}")
    print(f"Reporte generado: {args.output}")


if __name__ == "__main__":
    main()
