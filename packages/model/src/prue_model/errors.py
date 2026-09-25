"""Errores del dominio del modelo compartido."""


class PrueModelError(Exception):
    """Raiz de los errores de la libreria."""


class ModelNotFoundError(PrueModelError):
    """El nombre solicitado no esta en el registro.

    Deliberadamente NO hereda de KeyError: str(KeyError("x")) se renderiza
    como "'x'" y esas comillas terminarian en el detalle del 404.
    """


class DuplicateModelError(PrueModelError):
    """Se registro dos veces el mismo nombre."""


class InvalidFeaturesError(PrueModelError, ValueError):
    """El vector de entrada no cumple el contrato declarado por el spec."""


class ArtifactNotFoundError(PrueModelError, FileNotFoundError):
    """Falta el archivo .keras de un modelo registrado."""
