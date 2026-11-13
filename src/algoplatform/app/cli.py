"""Command line interface: ``python -m algoplatform <command>`` (or ``algoplatform <command>`` once installed).

Commands follow the platform's job groups:

==========================  ===========================================================================================
research and validation     ``strategies``, ``backtest``, ``optimize``, ``demo``
market data                 ``data scenarios | sample | providers | fetch | list | export | remove | sample-only``
platform                    ``structure``, ``functions``, ``web``
==========================  ===========================================================================================

Data for ``backtest`` / ``optimize``: ``--source sample`` (generated; ``--scenario`` picks a market scenario, otherwise
a mixed universe), ``--source history`` (bars downloaded with ``data fetch``) or ``--csv SYMBOL=file.csv``.

``data`` commands work on the same state directory as the web server (``--db-dir``, default ``./state``: history in
``platform.db``), through the same platform service, so the sample-data-only lock, key handling and audit apply the
same way. They act as the local administrator ``cli``: whoever can run them can read that directory anyway. Provider
keys come only from the environment (``ALPACA_API_KEY_ID`` / ``ALPACA_API_SECRET_KEY``; ``IB_HOST`` / ``IB_PORT`` /
``IB_CLIENT_ID``) and are never printed. A web server already running on the same directory sees new downloads after
a restart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from ..analytics.report import html_report, text_report, to_json
from ..backtest.engine import BacktestEngine
from ..core.config import PlatformConfig
from ..core.enums import Role
from ..core.exceptions import PlatformError
from ..core.logging import setup_logging
from ..data import MarketData, synthetic
from ..data.providers import TIMEFRAMES
from ..optimize import ParameterSpace, backtest_objective, optimize
from ..persistence import SqliteStore
from .. import fundamentals as _fundamentals, sentiment as _sentiment  # noqa: F401  (registers their strategies)
from ..strategy.base import STRATEGIES
from ..users.manager import User

DEFAULT_DB_DIR = "state"
OPTIMIZERS = ["grid", "random", "genetic", "bayesian", "tpe", "pso", "de", "annealing", "cmaes"]


# ---- helpers -------------------------------------------------------------------------------------------------------------
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


def _symbols(text: Optional[str]) -> List[str]:
    return [s.strip().upper() for s in (text or "").split(",") if s.strip()]


def _operator() -> User:
    """The local administrator the CLI acts as: actions pass the platform's permission checks and are audited as 'cli'."""
    return User("cli", "cli", password_hash="!", role=Role.ADMIN)


def _platform(db_dir: str, log_level: str = "WARNING"):
    """The platform on a state directory (the same files the web server uses with ``--db-dir``)."""
    from .platform import TradingPlatform                      # deferred: importing the platform wires every service
    os.makedirs(db_dir, exist_ok=True)
    return TradingPlatform(PlatformConfig(log_level=log_level), store=SqliteStore(os.path.join(db_dir, "platform.db")))


def _close(platform: Any) -> None:
    if platform is not None and platform.store is not None:
        platform.store.close()


def _load_data(args: argparse.Namespace) -> MarketData:
    """Market data for backtest / optimize, by ``--source`` (``--csv`` given means CSV files)."""
    md = MarketData()
    source = "csv" if args.csv else args.source
    if source == "csv":
        if not args.csv:
            raise SystemExit("--source csv needs --csv SYMBOL=file.csv [...]")
        for spec in args.csv:                     # SYMBOL=path.csv
            sym, _, path = spec.partition("=")
            md.add_csv(sym, path)
        return md
    if source == "history":
        platform = _platform(args.db_dir, args.log_level)
        try:
            wanted = _symbols(args.symbols) or platform.data.library.symbols("1Day")[:10]
            if not wanted:
                raise SystemExit(f"no downloaded history in {args.db_dir}: run 'data fetch' first")
            return platform.data.history(wanted, "1Day", min_rows=60)
        finally:
            _close(platform)
    symbols = _symbols(args.symbols) or ["AAA", "BBB", "CCC"]
    frames = (synthetic.scenario(args.scenario, symbols, n=args.bars, seed=args.seed) if args.scenario
              else synthetic.universe(symbols, n=args.bars, seed=args.seed))
    for s, df in frames.items():
        md.add(s, df)
    return md


def _table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return "(none)"
    cells = [[str(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = lambda vals: "  ".join(v.ljust(w) for v, w in zip(vals, widths))   # noqa: E731
    return "\n".join([line(cols), line(["-" * w for w in widths])] + [line(row) for row in cells])


def _print(obj: Any, as_json: bool, cols: Optional[List[str]] = None) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
    elif cols is not None:
        print(_table(obj, cols))
    else:
        print(obj)


# ---- parser ---------------------------------------------------------------------------------------------------------------
def _data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="sample", choices=["sample", "history", "csv"],
                   help="sample: generated data; history: bars downloaded with 'data fetch'; csv: --csv files")
    p.add_argument("--scenario", choices=sorted(synthetic.SCENARIOS), help="market scenario for sample data (default: a mixed universe)")
    p.add_argument("--symbols", default=None, help="comma list (sample default AAA,BBB,CCC; history default: everything downloaded)")
    p.add_argument("--csv", nargs="*", help="SYMBOL=file.csv (date,open,high,low,close,volume)")
    p.add_argument("--bars", type=int, default=756, help="bars of sample data")
    p.add_argument("--seed", type=int, default=7, help="seed of sample data")
    p.add_argument("--db-dir", default=DEFAULT_DB_DIR, help="state directory holding downloaded history (--source history)")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", default="sma_crossover", choices=STRATEGIES.names())
    _data_args(p)
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
    op.add_argument("--method", default="random", choices=OPTIMIZERS)
    op.add_argument("--n-iter", type=int, default=30)
    op.add_argument("--metric", default="sharpe")
    sub.add_parser("demo", help="end-to-end demo on sample data")

    data = sub.add_parser("data", help="market data: sample scenarios, providers, downloaded history, the sample-data-only lock")
    dsub = data.add_subparsers(dest="data_cmd", required=True)
    dsub.add_parser("scenarios", help="list the sample-data scenarios")
    sm = dsub.add_parser("sample", help="generate sample data, print its statistics, optionally write CSV files")
    sm.add_argument("--scenario", default="random_walk", choices=sorted(synthetic.SCENARIOS))
    sm.add_argument("--symbols", default="AAA,BBB,CCC")
    sm.add_argument("--bars", type=int, default=756)
    sm.add_argument("--seed", type=int, default=7)
    sm.add_argument("--timeframe", default="1Day", choices=list(TIMEFRAMES))
    sm.add_argument("--out", help="write SYMBOL.csv files into this directory")
    sm.add_argument("--json", action="store_true")
    pv = dsub.add_parser("providers", help="data providers configured by the environment (keys are never shown)")
    pv.add_argument("--test", action="store_true", help="fetch two weeks of SPY from each connected provider")
    pv.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    pv.add_argument("--json", action="store_true")
    fe = dsub.add_parser("fetch", help="download history from a provider into the history library")
    fe.add_argument("--provider", required=True, choices=["alpaca", "ib"])
    fe.add_argument("--symbols", required=True, help="comma list, up to 20")
    fe.add_argument("--start", help="first date (default: three years ago)")
    fe.add_argument("--end", help="last date (default: today)")
    fe.add_argument("--timeframe", default="1Day", choices=list(TIMEFRAMES))
    fe.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    fe.add_argument("--json", action="store_true")
    ls = dsub.add_parser("list", help="the downloaded series in the history library")
    ls.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    ls.add_argument("--json", action="store_true")
    ex = dsub.add_parser("export", help="write one downloaded series to a CSV file")
    ex.add_argument("symbol")
    ex.add_argument("--out", required=True)
    ex.add_argument("--timeframe", default="1Day", choices=list(TIMEFRAMES))
    ex.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    rm = dsub.add_parser("remove", help="remove one series from the history library")
    rm.add_argument("symbol")
    rm.add_argument("--timeframe", default="1Day", choices=list(TIMEFRAMES))
    rm.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    so = dsub.add_parser("sample-only", help="show or switch the platform-wide sample-data-only lock")
    so.add_argument("state", nargs="?", choices=["on", "off", "status"], default="status")
    so.add_argument("--db-dir", default=DEFAULT_DB_DIR)

    st = sub.add_parser("structure", help="show the job groups, their packages and roles, and verify the dependency rules")
    st.add_argument("--check", action="store_true", help="exit with status 1 if any dependency rule is violated")
    fn = sub.add_parser("functions", help="list the indicators, patterns and operators usable in formulas")
    fn.add_argument("--kind", choices=["indicator", "pattern", "operator"])
    web = sub.add_parser("web", help="start the web UI and JSON API")
    web.add_argument("--host", default="127.0.0.1", help="bind address (keep 127.0.0.1 unless behind TLS and an auth proxy)")
    web.add_argument("--port", type=int, default=8080)
    web.add_argument("--db-dir", default=None, help="keep users, sessions, approvals, alerts and history here across restarts")
    return ap


# ---- commands -------------------------------------------------------------------------------------------------------------
def _data_command(args: argparse.Namespace) -> int:
    c = args.data_cmd
    if c == "scenarios":
        for name, text in synthetic.SCENARIOS.items():
            print(f"{name:<20}{text}")
        return 0
    if c == "sample":
        frames = synthetic.scenario(args.scenario, _symbols(args.symbols), n=args.bars, seed=args.seed, timeframe=args.timeframe)
        rows = []
        for sym, df in frames.items():
            c0, c1 = float(df["close"].iloc[0]), float(df["close"].iloc[-1])
            rows.append({"symbol": sym, "bars": len(df), "from": str(df.index[0].date()), "to": str(df.index[-1].date()),
                         "first": round(c0, 2), "last": round(c1, 2), "return": f"{c1 / c0 - 1:+.1%}",
                         "worst_drawdown": f"{float((df['close'] / df['close'].cummax() - 1).min()):.1%}"})
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                df.rename_axis("date").to_csv(os.path.join(args.out, f"{sym}.csv"))
        _print(rows, args.json, ["symbol", "bars", "from", "to", "first", "last", "return", "worst_drawdown"])
        if args.out and not args.json:
            print(f"\nwrote {len(rows)} CSV file(s) to {args.out}")
        return 0

    platform = _platform(args.db_dir, args.log_level)
    try:
        svc, me = platform.data, _operator()
        if c == "providers":
            st = svc.status()
            rows = [{"provider": p["name"], "connected": "yes" if p["connected"] else "no",
                     "settings": ", ".join(f"{k}={v}" for k, v in p["settings"].items() if k in ("feed", "adjustment", "host", "port", "client_id"))}
                    for p in st["providers"]]
            if args.test:
                for row in rows:
                    if row["connected"] == "yes":
                        r = svc.test(me, row["provider"])
                        row["test"] = f"ok: {r['bars']} bars of {r['symbol']}" if r["ok"] else f"failed: {r['error']}"
            _print({"sample_only": st["sample_only"], "providers": rows, "environment": st["environment"]} if args.json else rows, args.json,
                   None if args.json else ["provider", "connected", "settings"] + (["test"] if args.test else []))
            if not args.json:
                print(f"\nsample-data-only lock: {'ON' if st['sample_only'] else 'off'}"
                      + ("" if any(r["connected"] == "yes" for r in rows) else
                         "\nno provider connected: set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY, or IB_HOST / IB_PORT"))
            return 0
        if c == "fetch":
            report = svc.fetch(me, args.provider, _symbols(args.symbols), args.start, args.end, args.timeframe)
            if args.json:
                _print(report, True)
            else:
                print(_table([{"symbol": s, "rows": v["rows"], "from": v["start"][:10], "to": v["end"][:10], "last_close": round(v["last_close"], 4)}
                              for s, v in report["stored"].items()], ["symbol", "rows", "from", "to", "last_close"]))
                for sym, why in report["errors"].items():
                    print(f"{sym}: {why}", file=sys.stderr)
            return 0 if report["stored"] else 2
        if c == "list":
            rows = svc.library.summary()
            _print(rows, args.json, None if args.json else ["symbol", "timeframe", "provider", "rows", "start", "end", "last_close"])
            return 0
        if c == "export":
            frame = svc.history([args.symbol.upper()], args.timeframe).frame(args.symbol.upper())
            frame.rename_axis("date").to_csv(args.out)
            print(f"wrote {len(frame)} bars of {args.symbol.upper()} to {args.out}")
            return 0
        if c == "remove":
            if not svc.remove_history(me, args.symbol.upper(), args.timeframe):
                print(f"{args.symbol.upper()} {args.timeframe} is not in the history library", file=sys.stderr)
                return 1
            print(f"removed {args.symbol.upper()} {args.timeframe}")
            return 0
        if c == "sample-only":
            if args.state != "status":
                svc.set_sample_only(me, args.state == "on")
                platform.save_state()
            print("sample-data-only lock: " + ("ON (downloads, downloaded history and uploads are refused)" if svc.sample_only else "off"))
            return 0
        raise SystemExit(f"unknown data command {c}")                 # argparse already restricts the choices
    finally:
        _close(platform)


def main(argv: Optional[List[str]] = None, web_runner: Optional[Callable[[argparse.Namespace], int]] = None) -> int:
    """``web_runner`` starts the web server for the ``web`` command; ``algoplatform/__main__.py`` supplies it, because the
    application layer may not import the web layer."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        if args.cmd == "strategies":
            for n in STRATEGIES.names():
                cls = STRATEGIES.get(n)
                print(f"{n:<20} params={cls.PARAMS}")
            return 0
        if args.cmd == "structure":
            from ..architecture import check_architecture, structure_text
            print(structure_text())
            problems = check_architecture()
            print("dependency rules: OK" if not problems else "dependency rules VIOLATED:\n  " + "\n  ".join(problems))
            return 1 if problems and args.check else 0
        if args.cmd == "functions":
            from ..technical.catalog import describe
            for f in describe(kind=args.kind):
                print(f"{f['kind']:<10}{f['category']:<12}{f['signature']}")
            return 0
        if args.cmd == "data":
            return _data_command(args)
        if args.cmd == "web":
            if web_runner is None:
                print("start the web UI with: python -m algoplatform web [--port 8080] [--db-dir ./state]", file=sys.stderr)
                return 2
            return web_runner(args)
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
            kwargs = {"n_iter": args.n_iter} if args.method in ("random", "bayesian", "tpe", "annealing") else {}
            out = optimize(obj, space, args.method, **kwargs)
            print(json.dumps(out.summary(), indent=2, default=str))
            print(out.top(5).to_string(index=False))
        return 0
    except PlatformError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
