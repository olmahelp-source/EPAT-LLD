"""Command line interface: ``python -m algoplatform <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from ..analytics.report import html_report, text_report, to_json
from ..backtest.engine import BacktestEngine
from ..core.config import PlatformConfig
from ..core.exceptions import PlatformError
from ..core.logging import setup_logging
from ..data import MarketData, synthetic
from ..optimize import ParameterSpace, backtest_objective, optimize
from .. import fundamentals as _fundamentals, sentiment as _sentiment  # noqa: F401  (registers their strategies)
from ..strategy.base import STRATEGIES


def _parse_params(items: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"bad --param '{it}', expected name=value")
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out


def _load_data(args: argparse.Namespace) -> MarketData:
    md = MarketData()
    if args.csv:
        for spec in args.csv:                     # SYMBOL=path.csv
            sym, _, path = spec.partition("=")
            md.add_csv(sym, path)
    else:
        symbols = args.symbols.split(",")
        for s, df in synthetic.universe(symbols, n=args.bars, seed=args.seed).items():
            md.add(s, df)
    return md


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", default="sma_crossover", choices=STRATEGIES.names())
    p.add_argument("--symbols", default="AAA,BBB,CCC", help="comma list (synthetic data)")
    p.add_argument("--csv", nargs="*", help="SYMBOL=file.csv (date,open,high,low,close,volume)")
    p.add_argument("--bars", type=int, default=756)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--param", action="append", help="strategy parameter name=value (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="algoplatform", description="Algorithmic trading platform")
    ap.add_argument("--log-level", default="WARNING")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("strategies", help="list built-in strategies")
    bt = sub.add_parser("backtest", help="run a backtest")
    _common(bt)
    bt.add_argument("--html", help="write an HTML tear sheet to this path")
    bt.add_argument("--json", action="store_true", help="print the report as JSON")
    op = sub.add_parser("optimize", help="optimise strategy parameters")
    _common(op)
    op.add_argument("--method", default="random", choices=["grid", "random", "genetic"])
    op.add_argument("--n-iter", type=int, default=30)
    op.add_argument("--metric", default="sharpe")
    sub.add_parser("demo", help="end-to-end demo on synthetic data")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        if args.cmd == "strategies":
            for n in STRATEGIES.names():
                cls = STRATEGIES.get(n)
                print(f"{n:<20} params={cls.PARAMS}")
            return 0
        if args.cmd == "demo":
            args = build_parser().parse_args(["backtest", "--strategy", "sma_crossover"])
        cfg = PlatformConfig(initial_capital=args.capital) if hasattr(args, "capital") else PlatformConfig()
        data = _load_data(args)
        params = _parse_params(args.param)
        symbols = data.symbols
        cls = STRATEGIES.get(args.strategy)
        if args.cmd == "backtest":
            res = BacktestEngine(data, cfg).run(cls(symbols, **params))
            rep = res.report()
            print(to_json(rep) if args.json else text_report(rep, f"{args.strategy} on {','.join(symbols)}"))
            if args.html:
                with open(args.html, "w", encoding="utf-8") as fh:
                    fh.write(html_report(res.equity, rep, args.strategy))
                print(f"\nHTML report written to {args.html}")
        else:
            space = ParameterSpace.from_bounds(cls.PARAM_BOUNDS)
            eng = BacktestEngine(data, cfg)
            obj = backtest_objective(eng, lambda **p: cls(symbols, **{**params, **p}), args.metric)
            kwargs = {"n_iter": args.n_iter} if args.method == "random" else {}
            out = optimize(obj, space, args.method, **kwargs)
            print(json.dumps(out.summary(), indent=2, default=str))
            print(out.top(5).to_string(index=False))
        return 0
    except PlatformError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
