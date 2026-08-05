import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import truss_analysis


def test_negative_validation_fuzz():
    errors = (
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        ZeroDivisionError,
    )
    skip_mods = ("main", "fileio", "postprocess")

    for _, modname, _ in pkgutil.walk_packages(
        truss_analysis.__path__, truss_analysis.__name__ + "."
    ):
        if modname.split(".")[-1] in skip_mods:
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        for _, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                for bad_arg in [None, 0, "", [], {}, 999.9]:
                    try:
                        obj(bad_arg)
                    except errors:
                        pass
