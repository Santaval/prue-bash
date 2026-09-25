"""Auto-importa cada modulo hermano para que el registro se llene solo.

Agregar un modelo es agregar un archivo en este paquete: no hay lista que
actualizar. Los modulos que empiezan con "_" se ignoran, lo que sirve de
valvula de escape para un modelo a medio escribir.
"""

import importlib
import pkgutil

__all__: list[str] = []


def _autoload() -> None:
    """Importa los modulos del paquete por su efecto secundario: registrarse."""
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module.name}")
        __all__.append(module.name)


_autoload()
