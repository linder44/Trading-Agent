"""Tests for onchain.py fixes: long/short ratio period fallback + funding timeout."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.onchain import OnChainAnalyzer


def _make_response(json_data, status_code=200, ok=True):
    """Create a mock Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.ok = ok
    return resp


class TestLongShortRatioFallback(unittest.TestCase):
    """Test that get_long_short_ratio tries multiple periods before giving up."""

    def setUp(self):
        self.analyzer = OnChainAnalyzer()
        self.mock_exchange = MagicMock()

    @patch("analysis.onchain.request_with_retry")
    def test_first_period_succeeds(self, mock_req):
        """5m period works — no fallback needed."""
        mock_req.return_value = _make_response({
            "data": [{"longAccountRatio": "0.6", "shortAccountRatio": "0.4"}]
        })

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["long_pct"], 60.0)
        self.assertEqual(result["short_pct"], 40.0)
        self.assertEqual(result["ratio"], 1.5)
        # Should only call once (5m worked)
        self.assertEqual(mock_req.call_count, 1)
        used_params = mock_req.call_args[1]["params"] if "params" in mock_req.call_args[1] else mock_req.call_args[0][1] if len(mock_req.call_args[0]) > 1 else mock_req.call_args.kwargs.get("params")
        # Actually check params from call_args
        call_kwargs = mock_req.call_args
        self.assertIn("5m", str(call_kwargs))

    @patch("analysis.onchain.request_with_retry")
    def test_fallback_to_second_period(self, mock_req):
        """5m returns None (400 error), 15m succeeds."""
        mock_req.side_effect = [
            None,  # 5m fails
            _make_response({
                "data": [{"longAccountRatio": "0.55", "shortAccountRatio": "0.45"}]
            }),  # 15m succeeds
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "ADA/USDT:USDT")

        self.assertEqual(result["long_pct"], 55.0)
        self.assertEqual(result["short_pct"], 45.0)
        self.assertEqual(mock_req.call_count, 2)

    @patch("analysis.onchain.request_with_retry")
    def test_fallback_to_third_period(self, mock_req):
        """5m and 15m fail, 1h succeeds."""
        mock_req.side_effect = [
            None,  # 5m fails
            None,  # 15m fails
            _make_response({
                "data": [{"longAccountRatio": "0.7", "shortAccountRatio": "0.3"}]
            }),  # 1h succeeds
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "DOGE/USDT:USDT")

        self.assertEqual(result["long_pct"], 70.0)
        self.assertEqual(result["ratio"], 2.33)
        self.assertEqual(result["signal"], "contrarian_bearish")
        self.assertEqual(mock_req.call_count, 3)

    @patch("analysis.onchain.request_with_retry")
    def test_all_periods_fail_ccxt_fallback(self, mock_req):
        """All REST periods fail, ccxt fallback succeeds."""
        mock_req.return_value = None  # All periods fail

        self.mock_exchange.exchange.fetch_long_short_ratio_history.return_value = [
            {"longAccount": 0.65, "shortAccount": 0.35}
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "UNI/USDT:USDT")

        self.assertEqual(result["long_pct"], 65.0)
        self.assertEqual(result["short_pct"], 35.0)
        self.assertEqual(mock_req.call_count, 3)  # Tried all 3 periods

    @patch("analysis.onchain.request_with_retry")
    def test_all_fail_returns_default(self, mock_req):
        """Everything fails — returns neutral default."""
        mock_req.return_value = None
        self.mock_exchange.exchange.fetch_long_short_ratio_history.side_effect = Exception("not supported")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BNB/USDT:USDT")

        self.assertEqual(result, {"long_pct": 50, "short_pct": 50, "ratio": 1.0, "signal": "neutral"})

    @patch("analysis.onchain.request_with_retry")
    def test_retries_set_to_1_per_period(self, mock_req):
        """Each period attempt uses retries=1 to avoid slow cascading retries."""
        mock_req.return_value = _make_response({
            "data": [{"longAccountRatio": "0.5", "shortAccountRatio": "0.5"}]
        })

        self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        # Verify retries=1 was passed
        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs.get("retries"), 1)

    @patch("analysis.onchain.request_with_retry")
    def test_cache_works_after_fallback(self, mock_req):
        """Result is cached after successful fallback — no extra API calls."""
        mock_req.side_effect = [
            None,  # 5m fails
            _make_response({
                "data": [{"longAccountRatio": "0.6", "shortAccountRatio": "0.4"}]
            }),  # 15m succeeds
        ]

        result1 = self.analyzer.get_long_short_ratio(self.mock_exchange, "NEAR/USDT:USDT")
        result2 = self.analyzer.get_long_short_ratio(self.mock_exchange, "NEAR/USDT:USDT")

        self.assertEqual(result1, result2)
        self.assertEqual(mock_req.call_count, 2)  # No extra calls for second request

    @patch("analysis.onchain.request_with_retry")
    def test_contrarian_bullish_signal(self, mock_req):
        """Low ratio produces contrarian_bullish signal."""
        mock_req.return_value = _make_response({
            "data": [{"longAccountRatio": "0.3", "shortAccountRatio": "0.7"}]
        })

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["signal"], "contrarian_bullish")

    @patch("analysis.onchain.request_with_retry")
    def test_empty_data_tries_next_period(self, mock_req):
        """Response with empty data array triggers next period."""
        mock_req.side_effect = [
            _make_response({"data": []}),  # 5m returns empty
            _make_response({
                "data": [{"longAccountRatio": "0.5", "shortAccountRatio": "0.5"}]
            }),  # 15m works
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "LTC/USDT:USDT")

        self.assertEqual(result["ratio"], 1.0)
        self.assertEqual(mock_req.call_count, 2)


class TestFundingRateTimeout(unittest.TestCase):
    """Test that funding rate uses increased timeout."""

    def setUp(self):
        self.analyzer = OnChainAnalyzer()
        self.mock_exchange = MagicMock()

    @patch("analysis.onchain.request_with_retry")
    def test_funding_rate_uses_15s_timeout(self, mock_req):
        """Funding rate request uses timeout=15 to avoid read timeouts."""
        mock_req.return_value = _make_response({
            "data": [{"fundingRate": "0.0001"}]
        })

        self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")

        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs.get("timeout"), 15)

    @patch("analysis.onchain.request_with_retry")
    def test_funding_rate_ccxt_fallback_on_timeout(self, mock_req):
        """When REST API returns None (timeout), ccxt fallback works."""
        mock_req.return_value = None
        self.mock_exchange.exchange.fetch_funding_rate.return_value = {
            "fundingRate": 0.0005
        }

        result = self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["funding_rate"], 0.0005)
        self.assertEqual(result["sentiment"], "bullish")

    @patch("analysis.onchain.request_with_retry")
    def test_funding_extreme_greed(self, mock_req):
        """High positive funding = extreme_greed."""
        mock_req.return_value = _make_response({
            "data": [{"fundingRate": "0.002"}]
        })

        result = self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["sentiment"], "extreme_greed")

    @patch("analysis.onchain.request_with_retry")
    def test_funding_extreme_fear(self, mock_req):
        """High negative funding = extreme_fear."""
        mock_req.return_value = _make_response({
            "data": [{"fundingRate": "-0.002"}]
        })

        result = self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["sentiment"], "extreme_fear")


if __name__ == "__main__":
    unittest.main()
