"""
vectorbt_bridge.py — convert executor order list to a vectorbt Portfolio.

This is the only custom glue needed between the executor's output and vectorbt's
analytics engine. Everything else (Sharpe, drawdown, alpha, benchmark comparison)
is handled by vbt.Portfolio methods directly.
"""

import vectorbt as vbt
import pandas as pd


def orders_to_portfolio(
    orders: list[dict],
    prices: pd.DataFrame,
    init_cash: float = 1_000_000.0,
) -> vbt.Portfolio:
    """
    Convert executor order list to a vectorbt Portfolio.

    Args:
        orders: List of order dicts from executor.main.run(simulate=True):
            [{"date": "2026-03-06", "ticker": "PLTR", "action": "ENTER",
              "shares": 100, "price_at_order": 84.12}, ...]
        prices: DataFrame indexed by date (datetime), columns by ticker.
                Build with price_loader.build_matrix().
        init_cash: Starting portfolio NAV.

    Returns:
        vbt.Portfolio with full analytics available via .sharpe_ratio(),
        .max_drawdown(), .total_return(), .plot(), etc.
    """
    tickers = prices.columns.tolist()
    dates = prices.index

    entries = pd.DataFrame(False, index=dates, columns=tickers)
    exits   = pd.DataFrame(False, index=dates, columns=tickers)
    sizes   = pd.DataFrame(0.0,   index=dates, columns=tickers)

    for order in orders:
        d = pd.Timestamp(order["date"])
        t = order["ticker"]
        if t not in tickers or d not in entries.index:
            continue
        if order["action"] == "ENTER":
            entries.loc[d, t] = True
            sizes.loc[d, t]   = float(order.get("shares", 0))
        elif order["action"] in ("EXIT", "REDUCE"):
            exits.loc[d, t] = True

    return vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        size=sizes,
        size_type="Amount",
        init_cash=init_cash,
        cash_sharing=True,
        group_by=True,
        fees=0.0,
        freq="D",
    )


def portfolio_stats(pf: vbt.Portfolio) -> dict:
    """
    Extract key metrics from a vectorbt Portfolio into a plain dict.

    Suitable for writing to metrics.json or printing as a summary.
    """
    return {
        "total_return": float(pf.total_return()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "calmar_ratio": float(pf.calmar_ratio()),
        "total_trades": int(pf.trades.count()),
        "win_rate": float(pf.trades.win_rate()),
    }
