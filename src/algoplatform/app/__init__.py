"""Application layer. Exports are lazy so low-level packages (e.g. backtest) can import
``app.engine`` without pulling in the facade, which itself depends on them."""
from typing import Any

__all__ = ["TradingEngine", "TradingPlatform"]


def __getattr__(name: str) -> Any:
    if name == "TradingEngine":
        from .engine import TradingEngine
        return TradingEngine
    if name == "TradingPlatform":
        from .platform import TradingPlatform
        return TradingPlatform
    raise AttributeError(name)
