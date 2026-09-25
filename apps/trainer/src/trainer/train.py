"""Entrena el modelo y escribe el artefacto en dist/."""

from pathlib import Path

import keras
import numpy as np
from prue_model import build_model

# La semilla fija hace que los pesos entrenados sean identicos entre corridas.
# Sin esto la cache de Turborepo devolveria modelos distintos segun si acerto.
# Ojo: los bytes del .keras igual varian entre corridas (es un zip con metadata),
# asi que no compares el artefacto por hash, compara los pesos.
SEED = 42
EPOCHS = 500

DIST_DIR = Path(__file__).resolve().parents[2] / "dist"
ARTIFACT = DIST_DIR / "math.keras"

# Relacion a aprender: y = 2x - 1
X_TRAIN = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y_TRAIN = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


def main() -> None:
    """Entrena el modelo y lo guarda."""
    keras.utils.set_random_seed(SEED)

    model = build_model()
    model.fit(X_TRAIN, Y_TRAIN, epochs=EPOCHS, verbose=0)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACT)
    print(f"Modelo guardado en {ARTIFACT}")


if __name__ == "__main__":
    main()
