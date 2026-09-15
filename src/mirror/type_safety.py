from importlib import import_module
from typing import Any, TypeVar, get_args, get_origin

from jsonargparse import Namespace

_PARAMS = ("RawT", "FormattedT", "BatchT")


def check_compatibility(cfg: Namespace) -> None:
    """Exit with a clear message if any selected subclasses (model, datasets,
    formatter, callbacks, ...) were parameterized with mutually incompatible
    generic types.
    """
    bindings = _bindings()
    seen: dict[str, tuple[Any, str]] = {}
    for selection in _selections(cfg):
        cls = _resolve_class(selection)
        for param, bound in _bound_params(cls, bindings).items():
            previous, owner = seen.setdefault(param, (bound, cls.__name__))
            if previous != bound:
                _fail(f"'{owner}' and '{cls.__name__}' disagree on {param} "
                      f"('{_name(previous)}' vs '{_name(bound)}')")


def _bindings() -> list[tuple[type, int]]:
    """Generic bases that constrain (RawT, FormattedT, BatchT), paired with how
    many leading type arguments map onto those params."""
    from mirror.callbacks.callback import Callback
    from mirror.datasets.mirror_dataset import MirrorDataset
    from mirror.formatters.mirror_formatter import MirrorFormatter
    from mirror.models.inference_model import InferenceModel
    from mirror.models.trainable_model import TrainableModel

    return [(TrainableModel, 3), (InferenceModel, 3),
            (MirrorFormatter, 3), (Callback, 3), (MirrorDataset, 1)]


def _bound_params(cls: type, bindings: list[tuple[type, int]]) -> dict[str, Any]:
    for base, count in bindings:
        args = _type_args(cls, base)
        if args is not None:
            return dict(zip(_PARAMS, args[:count]))
    return {}


def _selections(value: Any):
    """Yield every subclass selection (a Namespace with a class_path) reachable in
    the parsed config, descending into init_args, lists, and dicts."""
    if isinstance(value, Namespace):
        attrs = vars(value)
        if "class_path" in attrs:
            yield value
        for child in attrs.values():
            yield from _selections(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _selections(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _selections(child)


def _resolve_class(selection: Namespace) -> type:
    module_path, _, name = selection.class_path.rpartition(".")
    return getattr(import_module(module_path), name)


def _type_args(cls: type, base: type) -> tuple[Any, ...] | None:
    """Concrete type arguments `cls` supplies to generic `base`, or None when the
    relationship is absent or left generic (unresolved type variables)."""
    for klass in cls.__mro__:
        for origin in getattr(klass, "__orig_bases__", ()):
            if get_origin(origin) is base:
                args = get_args(origin)
                return None if any(isinstance(a, TypeVar) for a in args) else args
    return None


def _fail(message: str) -> None:
    raise SystemExit(f"Incompatible selection: {message}.")


def _name(t: Any) -> str:
    return getattr(t, "__name__", str(t))
