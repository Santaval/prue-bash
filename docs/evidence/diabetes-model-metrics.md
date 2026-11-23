# Métricas del modelo M07: riesgo de diabetes

- Artefacto: `apps/trainer/dist/diabetes.keras`
- Conjunto de prueba: `data/processed/diabetes_test.csv`
- Muestras: 19230
- Distribución real: clase 0 = 17534; clase 1 = 1696
- Umbral de clasificación: 0.5

## Resultados

| Métrica | Valor |
|---|---:|
| Accuracy | 0.9598 |
| Precision (diabetes) | 0.9138 |
| Recall (diabetes) | 0.6002 |
| F1 (diabetes) | 0.7246 |
| ROC-AUC | 0.9584 |

## Matriz de confusión

Filas = clase real; columnas = clase predicha.

| Real \ Predicha | No diabetes (0) | Diabetes (1) |
|---|---:|---:|
| No diabetes (0) | 17438 | 96 |
| Diabetes (1) | 678 | 1018 |

## Interpretación y límites

El conjunto está desbalanceado, por lo que accuracy por sí sola no describe el desempeño. Con el umbral configurado, recall mide la fracción de positivos detectados y precision la fracción de predicciones positivas que fueron correctas.

Estas métricas corresponden a un único split de prueba del dataset disponible. No constituyen validación clínica ni garantizan el mismo desempeño en otras poblaciones; la predicción no es un diagnóstico.
