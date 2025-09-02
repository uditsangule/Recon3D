from __future__ import annotations
from typing import Callable, Dict, Type, List, Optional
from importlib.metadata import entry_points

_ADAPTERS: Dict[str, Type] = {}
_METHODS: Dict[str, Type] = {}
_HOOKS: Dict[str, Type] = {}

class RegistryError(RuntimeError): pass

def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def register_adapter(name: Optional[str] = None):
    def deco(cls):
        n = _norm(name or getattr(cls, "name", cls.__name__))
        if n in _ADAPTERS and _ADAPTERS[n] is not cls:
            raise RegistryError(f"Adapter name conflict: '{n}'")
        _ADAPTERS[n] = cls
        return cls
    return deco

def register_method(name: Optional[str] = None):
    def deco(cls):
        n = _norm(name or getattr(cls, "name", cls.__name__))
        if n in _METHODS and _METHODS[n] is not cls:
            raise RegistryError(f"Method name conflict: '{n}'")
        _METHODS[n] = cls
        return cls
    return deco

def register_hook(name: Optional[str] = None):
    def deco(cls):
        n = _norm(name or getattr(cls, "name", cls.__name__))
        if n in _HOOKS and _HOOKS[n] is not cls:
            raise RegistryError(f"Hook name conflict: '{n}'")
        _HOOKS[n] = cls
        return cls
    return deco

def get_adapter(name: str):
    n = _norm(name)
    if n not in _ADAPTERS:
        raise RegistryError(f"Unknown adapter '{name}'. Available: {list(_ADAPTERS)}")
    return _ADAPTERS[n]

def get_method(name: str):
    n = _norm(name)
    if n not in _METHODS:
        raise RegistryError(f"Unknown method '{name}'. Available: {list(_METHODS)}")
    return _METHODS[n]

def get_hook(name: str):
    n = _norm(name)
    if n not in _HOOKS:
        raise RegistryError(f"Unknown hook '{name}'. Available: {list(_HOOKS)}")
    return _HOOKS[n]

def list_adapters() -> List[str]: return sorted(_ADAPTERS.keys())
def list_methods() -> List[str]:  return sorted(_METHODS.keys())
def list_hooks() -> List[str]:    return sorted(_HOOKS.keys())

# optional: auto-discovery via Python entry points
def load_entry_point_plugins(group: str = "src.plugins") -> int:
    """Discover and call plugin registrars declared in pyproject entry points.
    Each entry point should be a callable that, when imported/called, registers classes via decorators.
    """
    count = 0
    try:
        eps = entry_points(group=group)
    except Exception:
        return 0
    for ep in eps:
        try:
            registrar = ep.load()
            registrar()  # expected to call register_* inside
            count += 1
        except Exception:
            # swallow plugin errors to not break core; log in real code
            pass
    return count
