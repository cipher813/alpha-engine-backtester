"""Unit tests for loaders.price_loader — return structure, ffill, gap detection."""
import pytest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np


# ── load() return structure ──────────────────────────────────────────────────


class TestLoad:

    @patch("loaders.price_loader._load_from_s3")
    def test_s3_success_returns_ok_status(self, mock_s3):
        from loaders.price_loader import load

        mock_s3.return_value = {
            "date": "2026-03-28",
            "prices": {"AAPL": {"open": 150, "close": 152, "high": 153, "low": 149}},
        }
        result = load(bucket="test", price_date="2026-03-28")
        assert result["status"] == "ok"
        assert result["source"] == "s3"
        assert "AAPL" in result["prices"]

    @patch("loaders.price_loader._load_from_s3", side_effect=FileNotFoundError)
    def test_no_tickers_returns_no_data_status(self, mock_s3):
        from loaders.price_loader import load

        result = load(bucket="test", price_date="2026-03-28", tickers=None)
        assert result["status"] == "no_data"
        assert result["source"] == "none"
        assert result["prices"] == {}

    @patch("loaders.price_loader._load_from_s3", side_effect=FileNotFoundError)
    @patch("loaders.price_loader._load_from_yfinance")
    def test_yfinance_success_returns_ok(self, mock_yf, mock_s3):
        from loaders.price_loader import load

        mock_yf.return_value = {
            "date": "2026-03-28",
            "prices": {"AAPL": {"close": 152}},
        }
        result = load(bucket="test", price_date="2026-03-28", tickers=["AAPL"])
        assert result["status"] == "ok"
        assert result["source"] == "yfinance"

    @patch("loaders.price_loader._load_from_s3", side_effect=FileNotFoundError)
    @patch("loaders.price_loader._load_from_yfinance")
    def test_yfinance_empty_returns_no_data(self, mock_yf, mock_s3):
        from loaders.price_loader import load

        mock_yf.return_value = {"date": "2026-03-28", "prices": {}}
        result = load(bucket="test", price_date="2026-03-28", tickers=["AAPL"])
        assert result["status"] == "no_data"
        assert result["source"] == "none"


# ── build_matrix() ffill and gap detection ───────────────────────────────────


class TestBuildMatrix:

    @patch("loaders.price_loader._load_from_s3")
    def test_ffill_limited_to_5_days(self, mock_s3):
        """Tickers with gaps exceeding 5-day ffill limit are dropped entirely."""
        from loaders.price_loader import build_matrix

        # Simulate 10 dates where AAPL only has data on the first date
        dates = [f"2026-03-{d:02d}" for d in range(2, 12)]  # Mar 2-11

        def fake_s3(bucket, d, prefix):
            if d == "2026-03-02":
                return {"date": d, "prices": {"AAPL": {"close": 100.0}}}
            # Other dates have a different ticker but not AAPL
            return {"date": d, "prices": {"MSFT": {"close": 200.0}}}

        mock_s3.side_effect = fake_s3
        df = build_matrix(dates, bucket="test")

        # AAPL has 9 NaN days out of 10 — after ffill(limit=5) it still has
        # unfilled NaNs, so it gets dropped to avoid distorting VectorBT results
        assert "AAPL" not in df.columns, "AAPL should be dropped (gap > ffill limit)"
        assert "AAPL" in df.attrs.get("unfilled_gaps", {}), "AAPL should appear in unfilled_gaps"

    @patch("loaders.price_loader._load_from_s3")
    def test_gap_warnings_stored_in_attrs(self, mock_s3):
        """Price gap warnings should be stored in DataFrame attrs."""
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 12)]

        def fake_s3(bucket, d, prefix):
            # AAPL only on first date — will create > 5 day gap
            if d == "2026-03-02":
                return {"date": d, "prices": {"AAPL": {"close": 100.0}, "MSFT": {"close": 200.0}}}
            return {"date": d, "prices": {"MSFT": {"close": 200.0 + int(d[-2:])}}}

        mock_s3.side_effect = fake_s3
        df = build_matrix(dates, bucket="test")

        assert "price_gap_warnings" in df.attrs
        assert "unfilled_gaps" in df.attrs
        assert "staleness_warning" in df.attrs
        assert "stale_circuit_break" in df.attrs
        assert "no_data_dates" in df.attrs

    @patch("loaders.price_loader._load_from_s3")
    def test_empty_dates_returns_empty_df(self, mock_s3):
        """Empty dates list should return an empty DataFrame."""
        from loaders.price_loader import build_matrix

        df = build_matrix([], bucket="test")
        assert df.empty
