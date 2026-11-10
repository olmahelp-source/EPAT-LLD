"""Trade-level analytics and P&L attribution."""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from ..portfolio.portfolio import Trade
from ..utils.helpers import safe_div


def trades_frame(trades: Sequence[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["symbol", "direction", "quantity", "entry_price", "exit_price", "entry_ts",
                                     "exit_ts", "pnl", "commission", "strategy_id", "return_pct", "bars_held"])
    rows = []
    for t in trades:
        rows.append({"symbol": t.symbol, "direction": t.direction, "quantity": t.quantity,
                     "entry_price": t.entry_price, "exit_price": t.exit_price, "entry_ts": t.entry_ts,
                     "exit_ts": t.exit_ts, "pnl": t.pnl, "commission": t.commission, "strategy_id": t.strategy_id,
                     "return_pct": t.return_pct, "mae": t.mae, "mfe": t.mfe,
                     "hold_days": (t.holding_time.total_seconds() / 86400.0) if t.holding_time is not None else np.nan})
    return pd.DataFrame(rows)


def trade_statistics(trades: Sequence[Trade]) -> Dict[str, Any]:
    df = trades_frame(trades)
    if df.empty:
        return {"trades": 0}
    pnl = df["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    win_rate = float(len(wins) / len(pnl))
    # streaks
    signs = (pnl > 0).astype(int).tolist()
    max_w = max_l = cur_w = cur_l = 0
    for s in signs:
        cur_w, cur_l = (cur_w + 1, 0) if s else (0, cur_l + 1)
        max_w, max_l = max(max_w, cur_w), max(max_l, cur_l)
    top5 = pnl.nlargest(min(5, len(pnl))).sum()
    out = {
        "trades": int(len(df)), "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff_ratio": safe_div(avg_win, avg_loss), "expectancy": float(pnl.mean()),
        "profit_factor": safe_div(float(wins.sum()), float(-losses.sum()), default=math.inf if len(wins) else 0.0),
        "total_pnl": float(pnl.sum()), "best_trade": float(pnl.max()), "worst_trade": float(pnl.min()),
        "pnl_without_top5": float(pnl.sum() - top5),
        "avg_hold_days": float(df["hold_days"].mean()) if df["hold_days"].notna().any() else 0.0,
        "max_consecutive_wins": max_w, "max_consecutive_losses": max_l,
        "long_pnl": float(pnl[df["direction"] > 0].sum()), "short_pnl": float(pnl[df["direction"] < 0].sum()),
        "total_commission": float(df["commission"].sum()),
        "avg_mae_pct": float((df["mae"] / df["entry_price"]).mean()), "avg_mfe_pct": float((df["mfe"] / df["entry_price"]).mean()),
    }
    # exit efficiency: how much of the best excursion was captured
    best = (df["mfe"] * df["quantity"]).replace(0, np.nan)
    out["exit_efficiency"] = float(np.nanmean((df["pnl"] / best).clip(-1, 1))) if best.notna().any() else 0.0
    return out


def contribution_by(trades: Sequence[Trade], key: str = "symbol") -> pd.DataFrame:
    """P&L, count and win rate grouped by ``symbol``, ``strategy_id`` or ``direction``."""
    df = trades_frame(trades)
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(key)["pnl"]
    out = pd.DataFrame({"pnl": g.sum(), "trades": g.count(), "win_rate": g.apply(lambda s: (s > 0).mean()),
                        "avg_pnl": g.mean()})
    out["share_of_total"] = out["pnl"] / out["pnl"].sum() if out["pnl"].sum() != 0 else 0.0
    return out.sort_values("pnl", ascending=False)


def cost_attribution(gross_pnl: float, commission: float, slippage: float = 0.0) -> Dict[str, float]:
    net = gross_pnl - commission - slippage
    return {"gross_pnl": gross_pnl, "commission": commission, "slippage": slippage, "net_pnl": net,
            "cost_drag_pct_of_gross": safe_div(commission + slippage, abs(gross_pnl))}


def exposure_stats(positions_history: pd.DataFrame, equity: pd.Series) -> Dict[str, float]:
    """Average gross/net exposure and time-in-market from a frame of market values per symbol."""
    if positions_history.empty:
        return {"avg_gross": 0.0, "avg_net": 0.0, "time_in_market": 0.0}
    eq = equity.reindex(positions_history.index).ffill()
    gross = positions_history.abs().sum(axis=1) / eq
    net = positions_history.sum(axis=1) / eq
    return {"avg_gross": float(gross.mean()), "avg_net": float(net.mean()),
            "time_in_market": float((positions_history.abs().sum(axis=1) > 0).mean())}
