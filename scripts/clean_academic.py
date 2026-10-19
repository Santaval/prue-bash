"""Limpia la hoja 'Riesgo acad' (M09) y la separa en train/test.

Uso desde la raiz del repositorio:
    python scripts/clean_academic.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = ["attendance_pct", "previous_gpa", "study_hours_weekly",
           "failed_courses_history", "assignments_on_time_pct", "forum_participation_count",
           "is_working", "academic_risk_level"]

source = pd.read_excel("data/original/M07-M08-M09.xlsx", sheet_name="Riesgo acad")
data = source[COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
data = data[(data >= 0).all(axis=1)]

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
train, test = train_test_split(data, test_size=0.2, random_state=0)
train.to_csv(out / "academic_risk_train.csv", index=False)
test.to_csv(out / "academic_risk_test.csv", index=False)
