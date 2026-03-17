"""Tests for onchain.py fixes: long/short ratio period fallback + funding timeout.

Covers:
- HttpClientError (400) on account-long-short → skip remaining periods, cache symbol
- Period fallback (5m→15m→1h) on timeout/None
- ccxt fallback when all REST attempts fail
- Negative cache: unsupported symbol skips REST on second call
- Funding rate increased timeout (15s)
- All other onchain methods handle HttpClientError gracefully
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.onchain import OnChainAnalyzer
from utils.http import HttpClientError


def _make_response(json_data):
    """Create a mock Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = 200
    return resp


# ═══════════════════════════════════════════════════════════════════
# LONG/SHORT RATIO
# ═══════════════════════════════════════════════════════════════════

class TestLongShortRatio400Handling(unittest.TestCase):
    """Core fix: 400 on account-long-short means symbol is unsupported,
    don't try other periods, cache the failure."""

    def setUp(self):
        self.analyzer = OnChainAnalyzer()
        self.mock_exchange = MagicMock()

    @patch("analysis.onchain.request_with_retry")
    def test_400_skips_remaining_periods(self, mock_req):
        """When 5m returns 400, should NOT try 15m or 1h — go straight to ccxt."""
        mock_req.side_effect = HttpClientError(400, "Bad Request")
        self.mock_exchange.exchange.fetch_long_short_ratio_history.side_effect = Exception("nope")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BNB/USDT:USDT")

        # Only 1 REST call (5m), NOT 3
        self.assertEqual(mock_req.call_count, 1)
        self.assertEqual(result["ratio"], 1.0)
        self.assertEqual(result["signal"], "neutral")

    @patch("analysis.onchain.request_with_retry")
    def test_400_caches_unsupported_symbol(self, mock_req):
        """After 400, symbol is cached as unsupported — second call skips REST entirely."""
        mock_req.side_effect = HttpClientError(400, "Bad Request")
        self.mock_exchange.exchange.fetch_long_short_ratio_history.side_effect = Exception("nope")

        self.analyzer.get_long_short_ratio(self.mock_exchange, "BNB/USDT:USDT")
        mock_req.reset_mock()

        # Second call for same symbol — should NOT call REST at all
        self.analyzer._cache.clear()  # clear data cache to force re-fetch
        self.analyzer.get_long_short_ratio(self.mock_exchange, "BNB/USDT:USDT")

        self.assertEqual(mock_req.call_count, 0)  # zero REST calls
        self.assertIn("BNB", self.analyzer._unsupported_ls_symbols)

    @patch("analysis.onchain.request_with_retry")
    def test_400_then_ccxt_fallback_succeeds(self, mock_req):
        """400 on REST, but ccxt fallback returns data."""
        mock_req.side_effect = HttpClientError(400, "Bad Request")
        self.mock_exchange.exchange.fetch_long_short_ratio_history.return_value = [
            {"longAccount": 0.65, "shortAccount": 0.35}
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "ADA/USDT:USDT")

        self.assertEqual(result["long_pct"], 65.0)
        self.assertEqual(result["short_pct"], 35.0)
        self.assertEqual(mock_req.call_count, 1)

    @patch("analysis.onchain.request_with_retry")
    def test_different_symbols_tracked_independently(self, mock_req):
        """BNB gets 400, BTC works fine — they don't interfere."""
        def side_effect(url, params=None, **kwargs):
            sym = params.get("symbol", "") if params else ""
            if sym == "BNBUSDT":
                raise HttpClientError(400, "Bad Request")
            return _make_response({
                "data": [{"longAccountRatio": "0.6", "shortAccountRatio": "0.4"}]
            })

        mock_req.side_effect = side_effect
        self.mock_exchange.exchange.fetch_long_short_ratio_history.side_effect = Exception("nope")

        bnb = self.analyzer.get_long_short_ratio(self.mock_exchange, "BNB/USDT:USDT")
        btc = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(bnb["ratio"], 1.0)  # default
        self.assertEqual(btc["long_pct"], 60.0)  # real data
        self.assertIn("BNB", self.analyzer._unsupported_ls_symbols)
        self.assertNotIn("BTC", self.analyzer._unsupported_ls_symbols)


class TestLongShortRatioTimeoutFallback(unittest.TestCase):
    """Timeout (None) on one period → try next period."""

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
        self.assertEqual(result["ratio"], 1.5)
        self.assertEqual(mock_req.call_count, 1)

    @patch("analysis.onchain.request_with_retry")
    def test_timeout_fallback_to_next_period(self, mock_req):
        """5m times out (None), 15m succeeds."""
        mock_req.side_effect = [
            None,  # 5m timeout
            _make_response({
                "data": [{"longAccountRatio": "0.55", "shortAccountRatio": "0.45"}]
            }),
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "ETH/USDT:USDT")

        self.assertEqual(result["long_pct"], 55.0)
        self.assertEqual(mock_req.call_count, 2)

    @patch("analysis.onchain.request_with_retry")
    def test_all_timeouts_ccxt_fallback(self, mock_req):
        """All 3 periods timeout → ccxt fallback."""
        mock_req.return_value = None
        self.mock_exchange.exchange.fetch_long_short_ratio_history.return_value = [
            {"longAccount": 0.7, "shortAccount": 0.3}
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "SOL/USDT:USDT")

        self.assertEqual(result["long_pct"], 70.0)
        self.assertEqual(mock_req.call_count, 3)

    @patch("analysis.onchain.request_with_retry")
    def test_all_fail_returns_default(self, mock_req):
        """Everything fails — returns neutral default."""
        mock_req.return_value = None
        self.mock_exchange.exchange.fetch_long_short_ratio_history.side_effect = Exception("nope")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "XRP/USDT:USDT")

        self.assertEqual(result, {"long_pct": 50, "short_pct": 50, "ratio": 1.0, "signal": "neutral"})

    @patch("analysis.onchain.request_with_retry")
    def test_empty_data_tries_next_period(self, mock_req):
        """Response with empty data array triggers next period."""
        mock_req.side_effect = [
            _make_response({"data": []}),  # 5m: empty
            _make_response({
                "data": [{"longAccountRatio": "0.5", "shortAccountRatio": "0.5"}]
            }),
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "LTC/USDT:USDT")

        self.assertEqual(result["ratio"], 1.0)
        self.assertEqual(mock_req.call_count, 2)

    @patch("analysis.onchain.request_with_retry")
    def test_retries_1_per_period(self, mock_req):
        """Each period uses retries=1 for speed."""
        mock_req.return_value = _make_response({
            "data": [{"longAccountRatio": "0.5", "shortAccountRatio": "0.5"}]
        })

        self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs.get("retries"), 1)

    @patch("analysis.onchain.request_with_retry")
    def test_contrarian_bearish_signal(self, mock_req):
        mock_req.return_value = _make_response({
            "data": [{"longAccountRatio": "0.8", "shortAccountRatio": "0.2"}]
        })
        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")
        self.assertEqual(result["signal"], "contrarian_bearish")

    @patch("analysis.onchain.request_with_retry")
    def test_contrarian_bullish_signal(self, mock_req):
        mock_req.return_value = _make_response({
            "data": [{"longAccountRatio": "0.3", "shortAccountRatio": "0.7"}]
        })
        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")
        self.assertEqual(result["signal"], "contrarian_bullish")

    @patch("analysis.onchain.request_with_retry")
    def test_cache_works(self, mock_req):
        """Result cached — second call doesn't hit API."""
        mock_req.return_value = _make_response({
            "data": [{"longAccountRatio": "0.6", "shortAccountRatio": "0.4"}]
        })

        r1 = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")
        r2 = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(r1, r2)
        self.assertEqual(mock_req.call_count, 1)


# ═══════════════════════════════════════════════════════════════════
# FUNDING RATE
# ═══════════════════════════════════════════════════════════════════

class TestFundingRate(unittest.TestCase):

    def setUp(self):
        self.analyzer = OnChainAnalyzer()
        self.mock_exchange = MagicMock()

    @patch("analysis.onchain.request_with_retry")
    def test_uses_15s_timeout(self, mock_req):
        mock_req.return_value = _make_response({"data": [{"fundingRate": "0.0001"}]})

        self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")

        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs.get("timeout"), 15)

    @patch("analysis.onchain.request_with_retry")
    def test_ccxt_fallback_on_timeout(self, mock_req):
        mock_req.return_value = None
        self.mock_exchange.exchange.fetch_funding_rate.return_value = {"fundingRate": 0.0005}

        result = self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["funding_rate"], 0.0005)
        self.assertEqual(result["sentiment"], "bullish")

    @patch("analysis.onchain.request_with_retry")
    def test_400_handled_gracefully(self, mock_req):
        """HttpClientError from request_with_retry doesn't crash."""
        mock_req.side_effect = HttpClientError(400, "Bad Request")
        self.mock_exchange.exchange.fetch_funding_rate.return_value = {"fundingRate": 0.001}

        result = self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")

        self.assertIn("funding_rate", result)

    @patch("analysis.onchain.request_with_retry")
    def test_extreme_greed(self, mock_req):
        mock_req.return_value = _make_response({"data": [{"fundingRate": "0.002"}]})
        result = self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")
        self.assertEqual(result["sentiment"], "extreme_greed")

    @patch("analysis.onchain.request_with_retry")
    def test_extreme_fear(self, mock_req):
        mock_req.return_value = _make_response({"data": [{"fundingRate": "-0.002"}]})
        result = self.analyzer.get_funding_rate(self.mock_exchange, "BTC/USDT:USDT")
        self.assertEqual(result["sentiment"], "extreme_fear")


# ═══════════════════════════════════════════════════════════════════
# HTTP UTIL: HttpClientError
# ═══════════════════════════════════════════════════════════════════

class TestHttpClientError(unittest.TestCase):

    def test_is_exception(self):
        e = HttpClientError(400, "Bad Request")
        self.assertIsInstance(e, Exception)
        self.assertEqual(e.status_code, 400)

    @patch("utils.http.requests.get")
    def test_request_with_retry_raises_on_400(self, mock_get):
        """request_with_retry raises HttpClientError on 4xx."""
        from utils.http import request_with_retry

        resp = MagicMock()
        resp.status_code = 400
        resp.raise_for_status.side_effect = __import__("requests").exceptions.HTTPError(
            response=resp
        )
        mock_get.return_value = resp

        with self.assertRaises(HttpClientError) as ctx:
            request_with_retry("http://example.com/test", retries=1)

        self.assertEqual(ctx.exception.status_code, 400)

    @patch("utils.http.requests.get")
    def test_request_with_retry_returns_none_on_5xx(self, mock_get):
        """5xx errors still return None (server error, might work on retry)."""
        from utils.http import request_with_retry

        resp = MagicMock()
        resp.status_code = 500
        resp.response = resp
        http_err = __import__("requests").exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
        mock_get.return_value = resp

        result = request_with_retry("http://example.com/test", retries=1)

        self.assertIsNone(result)

    @patch("utils.http.requests.get")
    def test_request_with_retry_returns_response_on_200(self, mock_get):
        from utils.http import request_with_retry

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        mock_get.return_value = resp

        result = request_with_retry("http://example.com/test")

        self.assertEqual(result, resp)


# ═══════════════════════════════════════════════════════════════════
# OTHER ONCHAIN METHODS: HttpClientError handling
# ═══════════════════════════════════════════════════════════════════

class TestOtherMethodsHandle400(unittest.TestCase):
    """Ensure open_interest, whale_alerts, exchange_netflow don't crash on 400."""

    def setUp(self):
        self.analyzer = OnChainAnalyzer()
        self.mock_exchange = MagicMock()

    @patch("analysis.onchain.request_with_retry")
    def test_open_interest_400_falls_to_ccxt(self, mock_req):
        mock_req.side_effect = HttpClientError(400, "Bad Request")
        self.mock_exchange.exchange.fetch_open_interest.return_value = {
            "openInterestValue": 1000000, "openInterestAmount": 100
        }

        result = self.analyzer.get_open_interest(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["open_interest_amount"], 100)

    @patch("analysis.onchain.request_with_retry")
    def test_whale_alerts_400_falls_to_orderbook(self, mock_req):
        mock_req.side_effect = HttpClientError(400, "Bad Request")
        self.mock_exchange.exchange.fetch_order_book.return_value = {
            "bids": [[50000, 10]], "asks": [[50001, 1]]
        }

        result = self.analyzer.get_whale_alerts(self.mock_exchange)

        self.assertIsInstance(result, list)

    @patch("analysis.onchain.request_with_retry")
    def test_exchange_netflow_400_falls_to_ccxt(self, mock_req):
        mock_req.side_effect = HttpClientError(400, "Bad Request")
        self.mock_exchange.exchange.fetch_order_book.return_value = {
            "bids": [[50000, 10]], "asks": [[50001, 5]]
        }

        result = self.analyzer.get_exchange_netflow(self.mock_exchange)

        self.assertIn("signal", result)


if __name__ == "__main__":
    unittest.main()
