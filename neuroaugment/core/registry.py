from __future__ import annotations

from typing import Callable, Optional

_REGISTRY: dict[str, Callable] = {}


def register(name: Optional[str] = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name or fn.__name__] = fn
        return fn

    return decorator


def get_operator(name: str) -> Callable:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown operator '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_operators() -> list[str]:
    return sorted(_REGISTRY)


def build_operator_sequence(names: list[str]) -> list[Callable]:
    return [get_operator(name) for name in names]
