"""Generic name -> class registry used for strategies, brokers, sizers, optimizers."""
from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, Type, TypeVar

from .exceptions import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, Type[T]] = {}

    def register(self, name: str, *, overwrite: bool = False) -> Callable[[Type[T]], Type[T]]:
        key = name.lower()

        def deco(cls: Type[T]) -> Type[T]:
            if key in self._items and not overwrite and self._items[key] is not cls:
                raise ConfigError(f"{self.kind} '{name}' already registered")
            self._items[key] = cls
            return cls
        return deco

    def get(self, name: str) -> Type[T]:
        try:
            return self._items[name.lower()]
        except KeyError:
            raise ConfigError(f"unknown {self.kind} '{name}'",
                              details={"available": sorted(self._items)}) from None

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())
