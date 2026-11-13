"""InstrumentMaster: the single source of truth for what can be traded and what each broker calls it.

Why it exists: strategies, risk and analytics speak the platform's symbols; every broker has its own. The master owns
that translation in both directions, plus search, expiry handling and bulk loading with a per-row error report (one bad
row in a 50,000-line contract file must not block the other 49,999, and must not be silently dropped either).
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from ...foundation.core.exceptions import AggregateError, InstrumentError, MissingDataError
from ...foundation.core.models import Instrument
from ...foundation.utils.resilience import ErrorCollector
from .contracts import AssetClass, ContractSpec, _as_date


class InstrumentMaster:
    def __init__(self, specs: Iterable[ContractSpec] = ()) -> None:
        self._specs: Dict[str, ContractSpec] = {}
        self._by_broker: Dict[Tuple[str, str], str] = {}
        self._lock = threading.RLock()
        for s in specs:
            self.add(s)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._specs

    # ---- maintenance ----------------------------------------------------------------------------------------------------
    def add(self, spec: ContractSpec, overwrite: bool = False) -> ContractSpec:
        with self._lock:
            if spec.symbol in self._specs and not overwrite:
                raise InstrumentError(f"instrument '{spec.symbol}' already exists", code="DUPLICATE_INSTRUMENT")
            for broker, bsym in spec.broker_symbols:
                owner = self._by_broker.get((broker, bsym))
                if owner is not None and owner != spec.symbol:
                    raise InstrumentError(f"{broker} symbol '{bsym}' is already mapped to {owner}", code="BROKER_SYMBOL_CLASH")
            old = self._specs.get(spec.symbol)
            if old is not None:
                for key in old.broker_symbols:
                    self._by_broker.pop(key, None)
            self._specs[spec.symbol] = spec
            for key in spec.broker_symbols:
                self._by_broker[key] = spec.symbol
        return spec

    def deactivate(self, symbol: str) -> ContractSpec:
        """Mark as no longer tradable (delisted / expired) while keeping the record for history and reporting."""
        spec = self.get(symbol)
        return self.add(ContractSpec(**{**_fields(spec), "active": False}), overwrite=True)

    def load_records(self, records: Iterable[Mapping[str, Any]], overwrite: bool = False, on_error: str = "raise") -> Dict[str, Any]:
        """Bulk load. Every row is attempted. ``on_error="raise"`` raises one ``AggregateError`` listing all bad rows
        (nothing is loaded partially unnoticed); ``"skip"`` loads the good rows and returns the rejects in the report."""
        if on_error not in ("raise", "skip"):
            raise InstrumentError("on_error must be 'raise' or 'skip'")
        errors = ErrorCollector("loading instruments")
        staged: List[ContractSpec] = []
        for i, row in enumerate(records):
            with errors.catch(f"row {i + 1} ({row.get('symbol', '?')})"):
                staged.append(ContractSpec.from_dict(row))
        if errors.errors and on_error == "raise":
            errors.raise_if_any()
        loaded = 0
        for spec in staged:
            with errors.catch(spec.symbol):
                self.add(spec, overwrite)
                loaded += 1
        if errors.errors and on_error == "raise":
            raise AggregateError(f"loading instruments: {len(errors.errors)} row(s) rejected", [e for _, e in errors.errors],
                                 details={"rejected": errors.summary(), "loaded": loaded})
        return {"loaded": loaded, "rejected": errors.summary()}

    def load_csv(self, path: str, overwrite: bool = False, on_error: str = "raise") -> Dict[str, Any]:
        if not os.path.exists(path):
            raise MissingDataError(f"file not found: {path}")
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except (ValueError, pd.errors.ParserError) as exc:
            raise InstrumentError(f"cannot parse {path}: {exc}") from exc
        records: List[Dict[str, Any]] = [{str(k): v for k, v in row.items()} for row in df.to_dict("records")]
        return self.load_records(records, overwrite, on_error)

    # ---- lookup -----------------------------------------------------------------------------------------------------------
    def get(self, symbol: str) -> ContractSpec:
        try:
            return self._specs[symbol]
        except KeyError:
            raise InstrumentError(f"unknown instrument '{symbol}'", code="UNKNOWN_INSTRUMENT") from None

    def resolve(self, broker: str, broker_symbol: str) -> ContractSpec:
        """Broker's symbol -> platform record (used when fills and positions come back from a venue)."""
        key = (broker.lower(), broker_symbol)
        if key in self._by_broker:
            return self._specs[self._by_broker[key]]
        if broker_symbol in self._specs and broker.lower() not in dict(self._specs[broker_symbol].broker_symbols):
            return self._specs[broker_symbol]                       # no mapping recorded: the broker uses our symbol
        raise InstrumentError(f"{broker} symbol '{broker_symbol}' is not mapped to any instrument", code="UNMAPPED_SYMBOL")

    def broker_symbol(self, symbol: str, broker: str) -> str:
        return self.get(symbol).broker_symbol(broker)

    def search(self, text: str = "", asset_class: Optional[Any] = None, exchange: Optional[str] = None, sector: Optional[str] = None,
               underlying: Optional[str] = None, active_only: bool = True, asof: Optional[Any] = None) -> List[ContractSpec]:
        """Filter the master. ``text`` matches symbol, name or ISIN (case-insensitive); with ``asof`` expired contracts
        are left out."""
        ac = AssetClass(asset_class) if asset_class is not None and not isinstance(asset_class, AssetClass) else asset_class
        needle = text.lower()
        out = []
        for s in self._specs.values():
            if (active_only and not s.active) or (ac is not None and s.asset_class is not ac):
                continue
            if (exchange and s.exchange.lower() != exchange.lower()) or (sector and s.sector.lower() != sector.lower()):
                continue
            if (underlying and s.underlying != underlying) or (asof is not None and s.is_expired(asof)):
                continue
            if needle and needle not in s.symbol.lower() and needle not in s.name.lower() and needle != s.isin.lower():
                continue
            out.append(s)
        return sorted(out, key=lambda s: s.symbol)

    # ---- derivatives --------------------------------------------------------------------------------------------------------
    def expiries(self, underlying: str, asset_class: Any = AssetClass.OPTION, asof: Optional[Any] = None) -> List[Any]:
        return sorted({s.expiry for s in self.search(asset_class=asset_class, underlying=underlying, asof=asof) if s.expiry})

    def option_chain(self, underlying: str, expiry: Optional[Any] = None, asof: Optional[Any] = None) -> List[ContractSpec]:
        exp = _as_date(expiry)
        chain = [s for s in self.search(asset_class=AssetClass.OPTION, underlying=underlying, asof=asof) if exp is None or s.expiry == exp]
        return sorted(chain, key=lambda s: (s.expiry, s.strike, s.right))

    def front_future(self, root: str, asof: Any, roll_days: int = 0) -> ContractSpec:
        """Nearest unexpired future with more than ``roll_days`` left (the contract a continuous series would hold)."""
        live = [s for s in self.search(asset_class=AssetClass.FUTURE, underlying=root, asof=asof)
                if (s.days_to_expiry(asof) or 0) >= roll_days]
        if not live:
            raise InstrumentError(f"no live future for '{root}' as of {asof}", code="NO_FRONT_CONTRACT")
        return min(live, key=lambda s: s.expiry)                     # type: ignore[arg-type,return-value]

    # ---- export -------------------------------------------------------------------------------------------------------------
    def instruments(self, symbols: Optional[Iterable[str]] = None) -> Dict[str, Instrument]:
        """``{symbol: Instrument}`` for the engine / MarketData."""
        return {s: self.get(s).to_instrument() for s in (symbols if symbols is not None else list(self._specs))}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in sorted(self._specs.values(), key=lambda s: s.symbol)])

    def sectors(self) -> Dict[str, str]:
        return {s.symbol: s.sector for s in self._specs.values() if s.sector}

    # ---- persistence ----------------------------------------------------------------------------------------------------------
    def export_state(self) -> List[Dict[str, Any]]:
        """Every record as plain data (JSON-serialisable), for the store."""
        return [s.to_dict() for s in sorted(self._specs.values(), key=lambda s: s.symbol)]

    def import_state(self, records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        """Load records written by :meth:`export_state`, replacing any with the same symbol. A record that no longer
        validates is skipped and reported rather than blocking the start-up."""
        return self.load_records(records, overwrite=True, on_error="skip")


def _fields(spec: ContractSpec) -> Dict[str, Any]:
    return {name: getattr(spec, name) for name in spec.__dataclass_fields__}
