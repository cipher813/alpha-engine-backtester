"""Unit tests for loaders.signal_loader — dict-format signals handling."""
import pytest
from unittest.mock import patch, MagicMock

from loaders.signal_loader import load_buy_signals


class TestLoadBuySignals:

    @patch("loaders.signal_loader.load")
    def test_dict_format_signals_returned_as_list(self, mock_load):
        """Dict-keyed signals should be converted to list and filtered."""
        mock_load.return_value = {
            "signals": {
                "AAPL": {"ticker": "AAPL", "rating": "BUY", "score": 80},
                "MSFT": {"ticker": "MSFT", "rating": "HOLD", "score": 60},
                "NVDA": {"ticker": "NVDA", "rating": "BUY", "score": 90},
            }
        }
        result = load_buy_signals("test-bucket", "2026-03-29")
        assert len(result) == 2
        tickers = {s["ticker"] for s in result}
        assert tickers == {"AAPL", "NVDA"}

    @patch("loaders.signal_loader.load")
    def test_min_score_filter(self, mock_load):
        """min_score should filter out low-scoring BUY signals."""
        mock_load.return_value = {
            "signals": {
                "AAPL": {"ticker": "AAPL", "rating": "BUY", "score": 60},
                "NVDA": {"ticker": "NVDA", "rating": "BUY", "score": 90},
            }
        }
        result = load_buy_signals("test-bucket", "2026-03-29", min_score=70)
        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"

    @patch("loaders.signal_loader.load")
    def test_empty_signals_returns_empty(self, mock_load):
        """Empty signals dict should return empty list."""
        mock_load.return_value = {"signals": {}}
        result = load_buy_signals("test-bucket", "2026-03-29")
        assert result == []

    @patch("loaders.signal_loader.load")
    def test_missing_signals_key_returns_empty(self, mock_load):
        """Missing signals key should return empty list."""
        mock_load.return_value = {"date": "2026-03-29"}
        result = load_buy_signals("test-bucket", "2026-03-29")
        assert result == []
