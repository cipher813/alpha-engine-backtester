"""Tests for analysis/monte_carlo.py — materialized-label permutation test."""

import sqlite3
import tempfile

import pytest

from analysis.monte_carlo import run_monte_carlo


def _make_db(rows: list[tuple], horizon_cols: tuple[str, ...] = ("5d",)) -> str:
    """
    Build a temp research.db with score_performance populated from rows.
    Each row: (symbol, score_date, score, return_5d, spy_5d_return).
    Additional horizon_cols cause 10d/30d columns to be added (NULL).
    """
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    conn.execute(
        "CREATE TABLE score_performance ("
        "symbol TEXT, score_date TEXT, score REAL, price_on_date REAL, "
        "price_5d REAL, return_5d REAL, spy_5d_return REAL, beat_spy_5d INTEGER, "
        "price_10d REAL, return_10d REAL, spy_10d_return REAL, beat_spy_10d INTEGER, "
        "price_30d REAL, return_30d REAL, spy_30d_return REAL, beat_spy_30d INTEGER"
        ")"
    )
    conn.executemany(
        "INSERT INTO score_performance "
        "(symbol, score_date, score, return_5d, spy_5d_return) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return f.name


class TestInputValidation:
    def test_invalid_horizon(self):
        result = run_monte_carlo("/nonexistent/path.db", horizon="bogus")
        assert result["status"] == "error"
        assert "Invalid horizon" in result["error"]

    def test_missing_db(self):
        result = run_monte_carlo("/nonexistent/path.db")
        assert result["status"] == "error"
        assert "not found" in result["error"]


class TestInsufficientData:
    def test_empty_table(self):
        db = _make_db([])
        result = run_monte_carlo(db, n_permutations=10)
        assert result["status"] == "insufficient_data"

    def test_too_few_signals(self):
        rows = [
            ("AAPL", "2026-01-05", 75.0, 1.5, 0.5),
            ("MSFT", "2026-01-05", 80.0, 2.0, 0.5),
        ]
        db = _make_db(rows)
        result = run_monte_carlo(db, n_permutations=10)
        assert result["status"] == "insufficient_data"

    def test_all_below_min_score(self):
        rows = [
            (f"TICK{i}", f"2026-01-{(i % 28) + 1:02d}", 50.0, 1.0, 0.5)
            for i in range(30)
        ]
        db = _make_db(rows)
        result = run_monte_carlo(db, n_permutations=10, min_score=70.0)
        assert result["status"] == "insufficient_data"

    def test_too_few_signal_dates(self):
        # 30 signals but all on 3 dates → < 5 unique dates
        rows = []
        for i in range(10):
            rows.append((f"T{i}A", "2026-01-05", 80.0, 1.0, 0.5))
            rows.append((f"T{i}B", "2026-01-12", 80.0, 1.0, 0.5))
            rows.append((f"T{i}C", "2026-01-19", 80.0, 1.0, 0.5))
        db = _make_db(rows)
        result = run_monte_carlo(db, n_permutations=10)
        assert result["status"] == "insufficient_data"


class TestSignificantSignal:
    def test_high_score_beats_spy(self):
        """Synthetic: high scores have return > spy, low scores have return < spy.
        Top-N by score should cleanly beat the random (permuted) baseline."""
        rows = []
        # 10 dates, 10 tickers per date → 100 rows total
        for d in range(1, 11):
            date_str = f"2026-01-{d:02d}"
            for t in range(10):
                score = 90.0 if t < 3 else 72.0  # top-3 high-score
                stock_ret = 3.0 if t < 3 else 0.1  # top-3 outperform
                spy_ret = 0.5
                rows.append((f"T{t}", date_str, score, stock_ret, spy_ret))

        db = _make_db(rows)
        result = run_monte_carlo(db, n_permutations=200, top_n=3, min_score=70.0)

        assert result["status"] == "ok"
        assert result["actual_alpha"] > result["null_mean"]
        assert result["p_value"] < 0.05
        assert result["conclusion"] == "significant"
        assert result["n_signals"] == 100
        assert result["n_signal_dates"] == 10
        assert result["horizon"] == "5d"

    def test_deterministic_with_seed(self):
        """Same seed → identical p-values across runs."""
        rows = []
        for d in range(1, 11):
            for t in range(10):
                rows.append((
                    f"T{t}",
                    f"2026-01-{d:02d}",
                    75.0 + t,  # heterogeneous scores
                    (t - 5) * 0.3,  # heterogeneous returns
                    0.5,
                ))
        db = _make_db(rows)
        r1 = run_monte_carlo(db, n_permutations=50, seed=42)
        r2 = run_monte_carlo(db, n_permutations=50, seed=42)
        assert r1["actual_alpha"] == r2["actual_alpha"]
        assert r1["p_value"] == r2["p_value"]


class TestHorizonSelection:
    def test_10d_horizon_not_populated(self):
        """If the 10d columns are NULL, return insufficient_data."""
        rows = [
            (f"T{i}", f"2026-01-{(i % 28) + 1:02d}", 80.0, 1.0, 0.5)
            for i in range(30)
        ]
        db = _make_db(rows)
        result = run_monte_carlo(db, n_permutations=10, horizon="10d")
        assert result["status"] == "insufficient_data"
