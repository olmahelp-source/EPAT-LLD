"""The JSON API, one module per job group of ``architecture.py`` (Foundation has no routes: it is shared code).

``register_all`` adds every module's routes to the app. A test checks that the module names match the job groups and
that each page of the UI calls routes of its own group."""
from typing import Any, Dict

from . import access, application, capital, learning, market_data, operations, quant, signals, strategy, trading, validation

MODULES: Dict[str, Any] = {"access": access, "market_data": market_data, "signals": signals, "quant": quant, "learning": learning,
                           "strategy": strategy, "capital": capital, "trading": trading, "validation": validation,
                           "operations": operations, "application": application}


def register_all(app: Any) -> None:
    for module in MODULES.values():
        module.register(app)
