"""Limpia la hoja 'Monto Prestamo' (M08) y la separa en train/test.

Uso desde la raiz del repositorio:
    python scripts/clean_loan.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = ["applicant_income", "coapplicant_income", "credit_score",
           "years_employed", "owns_property", "loan_term_months",
           "loan_amount"]  # el objetivo va de ultimo

source = pd.read_excel("data/original/M07-M08-M09.xlsx", sheet_name="Monto Prestamo")
data = source[COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
data = data[(data >= 0).all(axis=1)]

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
train, test = train_test_split(data, test_size=0.2, random_state=42)
train.to_csv(out / "loan_train.csv", index=False)
test.to_csv(out / "loan_test.csv", index=False)
