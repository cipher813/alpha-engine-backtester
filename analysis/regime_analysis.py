"""
regime_analysis.py — split signal accuracy metrics by market_regime.

Joins score_performance with macro_snapshots (from research.db) to answer:
"Does signal quality vary meaningfully across bull/neutral/bear/caution regimes?"

Data availability: requires score_performance to be populated (Week 4+).
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from analysis.signal_quality import _compute_slice_metrics, MIN_SAMPLES

logger = logging.getLogger(__name__)


def load_with_regime(db_path: str) -> pd.DataFrame:
    """
    Load score_performance joined to macro_snapshots on score_date.

    Returns DataFrame with all score_performance columns plus market_regime.
    """
    path = Path(db_path).expanduser()
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            """
            SELECT sp.*, ms.market_regime
            FROM score_performance sp
            LEFT JOIN macro_snapshots ms ON date(sp.score_date) = ms.date
            ORDER BY sp.score_date
            """,
            conn,
            parse_dates=["score_date", "eval_date_10d", "eval_date_30d"],
        )
    finally:
        conn.close()

    logger.info(
        "Loaded %d score_performance rows with regime data (%d with regime populated)",
        len(df),
        df["market_regime"].notna().sum(),
    )
    return df


def accuracy_by_regime(df: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> list[dict]:
    """
    Compute accuracy metrics grouped by market_regime.

    Returns list of dicts, one per regime, each with the same structure as
    signal_quality._compute_slice_metrics().
    """
    populated_10d = df[df["beat_spy_10d"].notna()]
    populated_30d = df[df["beat_spy_30d"].notna()]

    if len(populated_10d) < min_samples:
        logger.warning(
            "Only %d rows with beat_spy_10d populated — regime analysis deferred until Week 4.",
            len(populated_10d),
        )
        return []

    regimes = populated_10d["market_regime"].dropna().unique()
    results = []

    for regime in sorted(regimes):
        slice_10d = populated_10d[populated_10d["market_regime"] == regime]
        slice_30d = populated_30d[populated_30d["market_regime"] == regime]
        metrics = _compute_slice_metrics(slice_10d, slice_30d)
        results.append({"market_regime": regime, **metrics})

    return results
