"""Typed, validated platform configuration."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, Optional

from .exceptions import ConfigError


@dataclass
class RiskConfig:
    max_order_quantity: float = 10_000
    max_order_notional: float = 250_000
    max_position_notional: float = 500_000      # per symbol
    max_gross_exposure_pct: float = 2.0         # of equity (2.0 = 200%)
    max_net_exposure_pct: float = 1.0
    price_collar_pct: float = 0.10              # reject limit prices >10% from last
    max_orders_per_minute: int = 120
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    max_consecutive_errors: int = 5
    max_data_staleness_sec: float = 60.0
    allow_short: bool = True
    restricted_symbols: tuple = ()


@dataclass
class SizingConfig:
    method: str = "fixed_fractional"
    risk_per_trade_pct: float = 0.01
    target_volatility: float = 0.15
    max_position_pct: float = 0.25              # of equity per symbol
    kelly_fraction: float = 0.5
    cash_buffer_pct: float = 0.02


@dataclass
class CostConfig:
    commission_per_share: float = 0.005
    commission_pct: float = 0.0
    min_commission: float = 1.0
    slippage_bps: float = 1.0
    spread_bps: float = 1.0
    impact_coef: float = 0.1                    # square-root-law coefficient
    transaction_tax_bps: float = 0.0            # e.g. STT on sells


@dataclass
class PlatformConfig:
    environment: str = "paper"                  # paper | live
    base_currency: str = "USD"
    initial_capital: float = 100_000.0
    log_level: str = "INFO"
    audit_path: Optional[str] = None
    risk: RiskConfig = field(default_factory=RiskConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.environment not in ("paper", "live"):
            raise ConfigError("environment must be 'paper' or 'live'")
        if self.initial_capital <= 0:
            raise ConfigError("initial_capital must be positive")
        r, s = self.risk, self.sizing
        for name in ("max_order_quantity", "max_order_notional", "max_position_notional"):
            if getattr(r, name) <= 0:
                raise ConfigError(f"risk.{name} must be positive")
        for name in ("max_daily_loss_pct", "max_drawdown_pct", "price_collar_pct"):
            if not 0 < getattr(r, name) <= 1:
                raise ConfigError(f"risk.{name} must be in (0, 1]")
        if not 0 < s.max_position_pct <= 1:
            raise ConfigError("sizing.max_position_pct must be in (0, 1]")
        if not 0 <= s.cash_buffer_pct < 1:
            raise ConfigError("sizing.cash_buffer_pct must be in [0, 1)")
        if not 0 < s.kelly_fraction <= 1:
            raise ConfigError("sizing.kelly_fraction must be in (0, 1]")

    # ---- (de)serialisation ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatformConfig":
        data = dict(data)
        sub = {"risk": RiskConfig, "sizing": SizingConfig, "costs": CostConfig}
        for key, klass in sub.items():
            if key in data:
                valid = {f.name for f in fields(klass)}
                bad = set(data[key]) - valid
                if bad:
                    raise ConfigError(f"unknown {key} settings: {sorted(bad)}")
                data[key] = klass(**data[key])
        valid_top = {f.name for f in fields(cls)}
        bad = set(data) - valid_top
        if bad:
            raise ConfigError(f"unknown settings: {sorted(bad)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "PlatformConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc

    @classmethod
    def from_env(cls, prefix: str = "ALGO_") -> "PlatformConfig":
        """Top-level scalar overrides from env vars, e.g. ALGO_INITIAL_CAPITAL=50000."""
        data: Dict[str, Any] = {}
        for f in fields(cls):
            raw = os.environ.get(prefix + f.name.upper())
            if raw is None or f.name in ("risk", "sizing", "costs"):
                continue
            try:
                data[f.name] = float(raw) if f.name == "initial_capital" else raw
            except ValueError as exc:
                raise ConfigError(f"bad value for {prefix}{f.name.upper()}: {raw}") from exc
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
