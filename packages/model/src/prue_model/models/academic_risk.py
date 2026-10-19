import numpy as np
import pandas as pd

from pathlib import Path
from keras import Sequential
from ..architecture import build_dense_classifier
from ..registry import register

_TRAIN = Path("data/processed/academic_risk_train.csv")
FEATURES = ("attendance_pct", "previous_gpa", "study_hours_weekly",
           "failed_courses_history", "assignments_on_time_pct", "forum_participation_count",
           "is_working")
CLASS_NAMES = ("0","1","2")

def dataset() -> tuple[np.ndarray, np.ndarray]:
    training_set = pd.read_csv(_TRAIN)

    input_train = training_set.drop(columns=["academic_risk_level"])
    expected_train = training_set["academic_risk_level"]

    return input_train, expected_train

@register (
    name = "academic_risk",
    input_features = len(FEATURES),
    feature_names=FEATURES,
    dataset=dataset,
    epochs=100,
    description="Riesgo academico",
    class_names=CLASS_NAMES
)


def build () -> Sequential:
    return build_dense_classifier(input_features=len(FEATURES), classes=len(CLASS_NAMES))