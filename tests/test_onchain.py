"""Tests for onchain.py: Bybit long/short ratio, funding timeout, HttpClientError.

Covers:
- Bybit v5 account-ratio as primary source for long/short ratio
- ccxt fallback when Bybit fails
- Default neutral when everything fails
- Funding rate: 15s timeout, ccxt fallback, sentiments
- Open interest: 15s timeout
- HttpClientError: 4xx vs 5xx distinction in request_with_retry
- Other onchain methods handle HttpClientError gracefully
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


def _bybit_ls_response(buy_ratio: str, sell_ratio: str, symbol: str = "BTCUSDT"):
    """Create mock Bybit v5 account-ratio response."""
    return _make_response({
        "retCode": 0,
        "result": {
            "list": [
                {"symbol": symbol, "buyRatio": buy_ratio,
                 "sellRatio": sell_ratio, "timestamp": "1234567890000"}
            ]
        }
    })


# ═══════════════════════════════════════════════════════════════════
# LONG/SHORT RATIO — BYBIT
# ═══════════════════════════════════════════════════════════════════

class TestLongShortRatioBybit(unittest.TestCase):
    """Bybit v5 account-ratio as primary source."""

    def setUp(self):
        self.analyzer = OnChainAnalyzer()
        self.mock_exchange = MagicMock()

    @patch("analysis.onchain.request_with_retry")
    def test_bybit_success_btc(self, mock_req):
        """BTC long/short ratio from Bybit."""
        mock_req.return_value = _bybit_ls_response("0.5162", "0.4838")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["long_pct"], 51.6)
        self.assertEqual(result["short_pct"], 48.4)
        self.assertEqual(result["ratio"], 1.07)
        self.assertEqual(result["signal"], "neutral")
        self.assertEqual(result["_source"], "bybit")
        # Verify Bybit URL was called
        call_url = mock_req.call_args[0][0]
        self.assertIn("api.bybit.com", call_url)
        self.assertIn("account-ratio", call_url)

    @patch("analysis.onchain.request_with_retry")
    def test_bybit_success_bnb(self, mock_req):
        """BNB works on Bybit (previously failed on Bitget with 400)."""
        mock_req.return_value = _bybit_ls_response("0.55", "0.45", "BNBUSDT")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BNB/USDT:USDT")

        self.assertEqual(result["long_pct"], 55.0)
        self.assertEqual(result["short_pct"], 45.0)
        self.assertEqual(result["ratio"], 1.22)

    @patch("analysis.onchain.request_with_retry")
    def test_bybit_success_doge(self, mock_req):
        """DOGE works on Bybit."""
        mock_req.return_value = _bybit_ls_response("0.60", "0.40", "DOGEUSDT")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "DOGE/USDT:USDT")

        self.assertEqual(result["long_pct"], 60.0)
        self.assertEqual(result["ratio"], 1.5)

    @patch("analysis.onchain.request_with_retry")
    def test_bybit_uses_correct_params(self, mock_req):
        """Verify correct Bybit API params."""
        mock_req.return_value = _bybit_ls_response("0.5", "0.5", "ADAUSDT")

        self.analyzer.get_long_short_ratio(self.mock_exchange, "ADA/USDT:USDT")

        _, kwargs = mock_req.call_args
        params = kwargs["params"]
        self.assertEqual(params["category"], "linear")
        self.assertEqual(params["symbol"], "ADAUSDT")
        self.assertEqual(params["period"], "1h")
        self.assertEqual(params["limit"], "1")

    @patch("analysis.onchain.request_with_retry")
    def test_bybit_uses_10s_timeout(self, mock_req):
        """Bybit request uses timeout=10."""
        mock_req.return_value = _bybit_ls_response("0.5", "0.5")

        self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs["timeout"], 10)

    @patch("analysis.onchain.request_with_retry")
    def test_contrarian_bearish(self, mock_req):
        """High ratio → contrarian_bearish signal."""
        mock_req.return_value = _bybit_ls_response("0.75", "0.25")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["signal"], "contrarian_bearish")

    @patch("analysis.onchain.request_with_retry")
    def test_contrarian_bullish(self, mock_req):
        """Low ratio → contrarian_bullish signal."""
        mock_req.return_value = _bybit_ls_response("0.3", "0.7")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["signal"], "contrarian_bullish")

    @patch("analysis.onchain.request_with_retry")
    def test_cache_prevents_duplicate_calls(self, mock_req):
        """Second call uses cache — no extra API call."""
        mock_req.return_value = _bybit_ls_response("0.6", "0.4")

        r1 = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")
        r2 = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(r1, r2)
        self.assertEqual(mock_req.call_count, 1)

    @patch("analysis.onchain.request_with_retry")
    def test_bybit_empty_list_falls_to_ccxt(self, mock_req):
        """Bybit returns empty list → ccxt fallback."""
        mock_req.return_value = _make_response({"retCode": 0, "result": {"list": []}})
        self.mock_exchange.exchange.fetch_long_short_ratio_history.return_value = [
            {"longAccount": 0.55, "shortAccount": 0.45}
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        self.assertEqual(result["long_pct"], 55.0)


class TestLongShortRatioFallback(unittest.TestCase):
    """Fallback behavior when Bybit is unavailable."""

    def setUp(self):
        self.analyzer = OnChainAnalyzer()
        self.mock_exchange = MagicMock()

    @patch("analysis.onchain.request_with_retry")
    def test_bybit_timeout_ccxt_fallback(self, mock_req):
        """Bybit times out (None) → ccxt fallback succeeds."""
        mock_req.return_value = None
        self.mock_exchange.exchange.fetch_long_short_ratio_history.return_value = [
            {"longAccount": 0.65, "shortAccount": 0.35}
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "SOL/USDT:USDT")

        self.assertEqual(result["long_pct"], 65.0)
        self.assertEqual(result["short_pct"], 35.0)
        self.assertEqual(result["_source"], "ccxt")

    @patch("analysis.onchain.request_with_retry")
    def test_bybit_400_ccxt_fallback(self, mock_req):
        """Bybit returns 400 → ccxt fallback."""
        mock_req.side_effect = HttpClientError(400, "Bad Request")
        self.mock_exchange.exchange.fetch_long_short_ratio_history.return_value = [
            {"longAccount": 0.6, "shortAccount": 0.4}
        ]

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "PEPE/USDT:USDT")

        self.assertEqual(result["long_pct"], 60.0)
        self.assertEqual(mock_req.call_count, 1)

    @patch("analysis.onchain.request_with_retry")
    def test_everything_fails_returns_neutral(self, mock_req):
        """Both Bybit and ccxt fail → neutral default."""
        mock_req.return_value = None
        self.mock_exchange.exchange.fetch_long_short_ratio_history.side_effect = Exception("nope")

        result = self.analyzer.get_long_short_ratio(self.mock_exchange, "XRP/USDT:USDT")

        self.assertEqual(result, {"long_pct": 50, "short_pct": 50, "ratio": 1.0, "signal": "neutral", "_source": "default"})

    @patch("analysis.onchain.request_with_retry")
    def test_no_bitget_or_binance_calls(self, mock_req):
        """Verify NO calls to Bitget or Binance."""
        mock_req.return_value = _bybit_ls_response("0.5", "0.5")

        self.analyzer.get_long_short_ratio(self.mock_exchange, "BTC/USDT:USDT")

        call_url = mock_req.call_args[0][0]
        self.assertNotIn("bitget", call_url)
        self.assertNotIn("binance", call_url)
        self.assertIn("bybit", call_url)


class TestParseBybitLs(unittest.TestCase):
    """Unit tests for _parse_bybit_ls helper."""

    def test_normal_data(self):
        item = {"buyRatio": "0.5162", "sellRatio": "0.4838", "timestamp": "123"}
        result = OnChainAnalyzer._parse_bybit_ls(item)
        self.assertEqual(result["long_pct"], 51.6)
        self.assertEqual(result["short_pct"], 48.4)
        self.assertEqual(result["ratio"], 1.07)
        self.assertEqual(result["signal"], "neutral")
        self.assertEqual(result["_source"], "bybit")

    def test_extreme_long(self):
        item = {"buyRatio": "0.85", "sellRatio": "0.15"}
        result = OnChainAnalyzer._parse_bybit_ls(item)
        self.assertEqual(result["signal"], "contrarian_bearish")

    def test_extreme_short(self):
        item = {"buyRatio": "0.2", "sellRatio": "0.8"}
        result = OnChainAnalyzer._parse_bybit_ls(item)
        self.assertEqual(result["signal"], "contrarian_bullish")

    def test_equal_split(self):
        item = {"buyRatio": "0.5", "sellRatio": "0.5"}
        result = OnChainAnalyzer._parse_bybit_ls(item)
        self.assertEqual(result["ratio"], 1.0)
        self.assertEqual(result["signal"], "neutral")


# ═══════════════════════════════════════════════════════════════════
# OPEN INTEREST TIMEOUT
# ═══════════════════════════════════════════════════════════════════

class TestOpenInterestTimeout(unittest.TestCase):
    """Open interest uses 15s timeout (not default 5s)."""

    def setUp(self):
        self.analyzer = OnChainAnalyzer()
        self.mock_exchange = MagicMock()

    @patch("analysis.onchain.request_with_retry")
    def test_uses_15s_timeout(self, mock_req):
        mock_req.return_value = _make_response({"data": {"openInterestList": [{"size": "100"}]}})
        self.mock_exchange.exchange.fetch_ticker.return_value = {"last": 50000}
        self.analyzer.get_open_interest(self.mock_exchange, "BTC/USDT:USDT")
        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs.get("timeout"), 15)


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
    def test_raises_on_400(self, mock_get):
        from utils.http import request_with_retry
        resp = MagicMock()
        resp.status_code = 400
        resp.raise_for_status.side_effect = __import__("requests").exceptions.HTTPError(response=resp)
        mock_get.return_value = resp

        with self.assertRaises(HttpClientError) as ctx:
            request_with_retry("http://example.com/test", retries=1)
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("utils.http.requests.get")
    def test_returns_none_on_5xx(self, mock_get):
        from utils.http import request_with_retry
        resp = MagicMock()
        resp.status_code = 500
        resp.response = resp
        resp.raise_for_status.side_effect = __import__("requests").exceptions.HTTPError(response=resp)
        mock_get.return_value = resp

        result = request_with_retry("http://example.com/test", retries=1)
        self.assertIsNone(result)

    @patch("utils.http.requests.get")
    def test_returns_response_on_200(self, mock_get):
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
