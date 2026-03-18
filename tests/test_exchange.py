"""Tests for exchange client — trigger orders and SL/TP orders.

Verifies that Bitget-specific parameters are passed correctly to ccxt,
particularly triggerType values that caused error 400172,
and SL/TP params for plan orders (stopSurplusTriggerType fix for error 40017).

Tests use TWO levels of verification:
1. Unit tests: check that ExchangeClient passes correct params to ccxt
2. Integration tests: use REAL ccxt bitget mapping to verify final Bitget API params
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_client(mock_exchange):
    """Helper: create ExchangeClient with mocked ccxt exchange."""
    with patch("exchange.client.ccxt.bitget") as mock_bitget:
        mock_bitget.return_value = mock_exchange
        mock_exchange.markets = {
            "BTC/USDT:USDT": {"swap": True},
            "ETH/USDT:USDT": {"swap": True},
            "SOL/USDT:USDT": {"swap": True},
        }
        from config import BitgetConfig
        cfg = BitgetConfig()
        cfg.api_key = ""
        cfg.secret_key = ""
        cfg.passphrase = ""
        cfg.demo = False
        from exchange.client import ExchangeClient
        return ExchangeClient(cfg)


# ── Test: create_trigger_order (without SL/TP) ──────────────────────


class TestCreateTriggerOrder(unittest.TestCase):
    """Test create_trigger_order sends correct params to Bitget."""

    def setUp(self):
        self.mock_exchange = MagicMock()
        self.mock_exchange.create_order.return_value = {"id": "order123"}
        self.client = _make_client(self.mock_exchange)

    def _get_params(self):
        args, kwargs = self.mock_exchange.create_order.call_args
        return args[4] if len(args) > 4 else kwargs.get("params", {})

    def test_trigger_order_uses_mark_price(self):
        """triggerType must be 'mark_price', not 'market_price' (Bitget 400172 fix)."""
        self.client.create_trigger_order("BTC/USDT:USDT", "buy", 0.01, 95000.0)
        params = self._get_params()
        self.assertEqual(params["triggerType"], "mark_price")
        self.assertEqual(params["triggerPrice"], 95000.0)

    def test_trigger_order_not_market_price(self):
        """Ensure 'market_price' (the old buggy value) is NOT used."""
        self.client.create_trigger_order("ETH/USDT:USDT", "sell", 0.1, 3500.0)
        params = self._get_params()
        self.assertNotEqual(params["triggerType"], "market_price")

    def test_trigger_order_returns_order(self):
        expected = {"id": "order789", "status": "open"}
        self.mock_exchange.create_order.return_value = expected
        result = self.client.create_trigger_order("BTC/USDT:USDT", "buy", 0.05, 90000.0)
        self.assertEqual(result, expected)

    def test_trigger_order_with_extra_params(self):
        self.client.create_trigger_order(
            "BTC/USDT:USDT", "buy", 0.01, 95000.0,
            params={"reduceOnly": True},
        )
        params = self._get_params()
        self.assertEqual(params["triggerType"], "mark_price")
        self.assertTrue(params["reduceOnly"])

    def test_trigger_order_sell_side(self):
        self.client.create_trigger_order("BTC/USDT:USDT", "sell", 0.02, 110000.0)
        args, _ = self.mock_exchange.create_order.call_args
        self.assertEqual(args[2], "sell")
        params = self._get_params()
        self.assertEqual(params["triggerType"], "mark_price")


# ── Test: create_market_order_with_sltp ──────────────────────────────


class TestCreateMarketOrderWithSLTP(unittest.TestCase):
    """Test create_market_order_with_sltp sends correct SL/TP params."""

    def setUp(self):
        self.mock_exchange = MagicMock()
        self.mock_exchange.create_order.return_value = {"id": "sltp_order"}
        self.client = _make_client(self.mock_exchange)

    def _get_params(self):
        args, kwargs = self.mock_exchange.create_order.call_args
        return args[4] if len(args) > 4 else kwargs.get("params", {})

    def test_sltp_params_structure(self):
        """SL/TP must use ccxt unified dict format with triggerPrice and type."""
        self.client.create_market_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 94000.0, 100000.0,
        )
        params = self._get_params()
        self.assertIn("stopLoss", params)
        self.assertIn("takeProfit", params)
        self.assertEqual(params["stopLoss"]["triggerPrice"], 94000.0)
        self.assertEqual(params["stopLoss"]["type"], "market")
        self.assertEqual(params["takeProfit"]["triggerPrice"], 100000.0)
        self.assertEqual(params["takeProfit"]["type"], "market")

    def test_sltp_order_type_is_market(self):
        self.client.create_market_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 94000.0, 100000.0,
        )
        args, _ = self.mock_exchange.create_order.call_args
        self.assertEqual(args[1], "market")

    def test_sltp_no_triggerPrice(self):
        """Regular market orders must NOT have triggerPrice (that makes it a plan order)."""
        self.client.create_market_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 94000.0, 100000.0,
        )
        params = self._get_params()
        self.assertNotIn("triggerPrice", params)


# ── Test: create_trigger_order_with_sltp (unit level) ────────────────


class TestTriggerOrderWithSLTPUnit(unittest.TestCase):
    """Test create_trigger_order_with_sltp passes correct ccxt unified params.

    The key insight: ccxt handles the conversion from unified params to
    Bitget-native fields. We must use ccxt's 'stopLoss'/'takeProfit' dicts,
    NOT raw Bitget fields like presetStopLossPrice or stopSurplusTriggerType.
    """

    def setUp(self):
        self.mock_exchange = MagicMock()
        self.mock_exchange.create_order.return_value = {"id": "trig_sltp"}
        self.client = _make_client(self.mock_exchange)

    def _get_params(self):
        args, kwargs = self.mock_exchange.create_order.call_args
        return args[4] if len(args) > 4 else kwargs.get("params", {})

    def test_has_trigger_price(self):
        self.client.create_trigger_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 95000.0, 93000.0, 99000.0,
        )
        params = self._get_params()
        self.assertEqual(params["triggerPrice"], 95000.0)

    def test_has_trigger_type_mark_price(self):
        self.client.create_trigger_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 95000.0, 93000.0, 99000.0,
        )
        params = self._get_params()
        self.assertEqual(params["triggerType"], "mark_price")

    def test_uses_ccxt_stopLoss_dict(self):
        """Must use ccxt unified stopLoss dict — ccxt maps it to Bitget fields."""
        self.client.create_trigger_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 95000.0, 93000.0, 99000.0,
        )
        params = self._get_params()
        self.assertIn("stopLoss", params)
        self.assertIsInstance(params["stopLoss"], dict)
        self.assertEqual(params["stopLoss"]["triggerPrice"], 93000.0)
        self.assertEqual(params["stopLoss"]["type"], "mark_price")

    def test_uses_ccxt_takeProfit_dict(self):
        """Must use ccxt unified takeProfit dict — ccxt maps it to Bitget fields."""
        self.client.create_trigger_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 95000.0, 93000.0, 99000.0,
        )
        params = self._get_params()
        self.assertIn("takeProfit", params)
        self.assertIsInstance(params["takeProfit"], dict)
        self.assertEqual(params["takeProfit"]["triggerPrice"], 99000.0)
        self.assertEqual(params["takeProfit"]["type"], "mark_price")

    def test_no_raw_bitget_params(self):
        """Must NOT use raw Bitget params — they bypass ccxt mapping and cause 40017."""
        self.client.create_trigger_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 95000.0, 93000.0, 99000.0,
        )
        params = self._get_params()
        # These raw Bitget params caused error 40017
        self.assertNotIn("presetStopLossPrice", params)
        self.assertNotIn("presetStopLossTriggerType", params)
        self.assertNotIn("presetStopSurplusPrice", params)
        self.assertNotIn("presetStopSurplusTriggerType", params)
        # These are plan-order native fields — ccxt should set them, not us
        self.assertNotIn("stopLossTriggerPrice", params)
        self.assertNotIn("stopLossTriggerType", params)
        self.assertNotIn("stopSurplusTriggerPrice", params)
        self.assertNotIn("stopSurplusTriggerType", params)

    def test_sell_side_short(self):
        self.client.create_trigger_order_with_sltp(
            "ETH/USDT:USDT", "sell", 0.5, 3500.0, 3700.0, 3200.0,
        )
        args, kwargs = self.mock_exchange.create_order.call_args
        self.assertEqual(args[2], "sell")
        params = self._get_params()
        self.assertEqual(params["stopLoss"]["triggerPrice"], 3700.0)
        self.assertEqual(params["takeProfit"]["triggerPrice"], 3200.0)

    def test_returns_order(self):
        expected = {"id": "xyz", "status": "open"}
        self.mock_exchange.create_order.return_value = expected
        result = self.client.create_trigger_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 95000.0, 93000.0, 99000.0,
        )
        self.assertEqual(result, expected)


# ── Test: REAL ccxt mapping (integration with actual ccxt bitget) ─────


class TestCcxtBitgetPlanOrderMapping(unittest.TestCase):
    """Integration test: verify that ccxt ACTUALLY maps our params to correct Bitget fields.

    This uses a real ccxt.bitget instance to verify the final API request body,
    ensuring our params produce the right Bitget API fields:
    - stopLossTriggerPrice (NOT presetStopLossPrice)
    - stopLossTriggerType (NOT presetStopLossTriggerType)
    - stopSurplusTriggerPrice (NOT presetStopSurplusPrice)
    - stopSurplusTriggerType (NOT presetStopSurplusTriggerType)
    """

    # Full market definition required by ccxt.bitget internals
    _MARKET = {
        "id": "BTCUSDT",
        "symbol": "BTC/USDT:USDT",
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
        "settleId": "USDT",
        "type": "swap",
        "spot": False,
        "margin": False,
        "swap": True,
        "future": False,
        "option": False,
        "linear": True,
        "inverse": False,
        "contract": True,
        "contractSize": 0.001,
        "precision": {"amount": 0.001, "price": 0.1},
        "limits": {
            "amount": {"min": 0.001, "max": 1000},
            "price": {"min": 0.01, "max": 999999},
            "cost": {"min": None, "max": None},
        },
        "info": {"symbolName": "BTCUSDT"},
        "active": True,
    }

    def setUp(self):
        """Create a real ccxt.bitget instance and load minimal markets."""
        import ccxt
        self.exchange = ccxt.bitget({
            "apiKey": "testkey123",
            "secret": "testsecret123456789012345678901234567890",
            "password": "testpass",
        })
        self.exchange.markets = {"BTC/USDT:USDT": self._MARKET}
        self.exchange.markets_by_id = {"BTCUSDT": self._MARKET}

    def _build_request(self, params):
        """Build the request body that ccxt would send to Bitget.

        Intercepts the HTTP fetch call to capture the actual JSON body
        that ccxt produces after all its internal param mapping.
        Returns the parsed body dict.
        """
        import json as _json
        captured = {"body": ""}

        def mock_fetch(url, method="GET", headers=None, body=None):
            captured["body"] = body or ""
            # Return a fake successful Bitget response
            return _json.dumps({
                "code": "00000",
                "msg": "success",
                "requestTime": 0,
                "data": {"orderId": "fake123", "clientOid": ""},
            })

        self.exchange.fetch = mock_fetch

        try:
            self.exchange.create_order(
                "BTC/USDT:USDT", "market", "buy", 0.01, params=params,
            )
        except Exception:
            pass

        body_str = captured["body"]
        return _json.loads(body_str) if body_str else {}

    def _plan_order_params(self):
        """Standard params for a plan order with SL/TP."""
        return {
            "triggerPrice": 95000.0,
            "triggerType": "mark_price",
            "stopLoss": {"triggerPrice": 93000.0, "type": "mark_price"},
            "takeProfit": {"triggerPrice": 99000.0, "type": "mark_price"},
        }

    def test_plan_order_with_sltp_has_stopLossTriggerType(self):
        """ccxt must produce stopLossTriggerType in the final Bitget request."""
        body = self._build_request(self._plan_order_params())
        self.assertIn("stopLossTriggerType", body,
                       "ccxt must produce stopLossTriggerType for plan orders")
        self.assertEqual(body["stopLossTriggerType"], "mark_price")

    def test_plan_order_with_sltp_has_stopSurplusTriggerType(self):
        """ccxt must produce stopSurplusTriggerType — the field from error 40017."""
        body = self._build_request(self._plan_order_params())
        self.assertIn("stopSurplusTriggerType", body,
                       "ccxt must produce stopSurplusTriggerType for plan orders")
        self.assertEqual(body["stopSurplusTriggerType"], "mark_price")

    def test_plan_order_with_sltp_has_stopLossTriggerPrice(self):
        """ccxt must produce stopLossTriggerPrice with correct value."""
        body = self._build_request(self._plan_order_params())
        self.assertIn("stopLossTriggerPrice", body)
        self.assertIn("93000", str(body["stopLossTriggerPrice"]))

    def test_plan_order_with_sltp_has_stopSurplusTriggerPrice(self):
        """ccxt must produce stopSurplusTriggerPrice with correct value."""
        body = self._build_request(self._plan_order_params())
        self.assertIn("stopSurplusTriggerPrice", body)
        self.assertIn("99000", str(body["stopSurplusTriggerPrice"]))

    def test_plan_order_NOT_preset_fields(self):
        """Plan orders must NOT use preset* fields (those are for regular orders)."""
        body = self._build_request(self._plan_order_params())
        self.assertNotIn("presetStopLossPrice", body,
                         "preset* fields are for regular orders, not plan orders")
        self.assertNotIn("presetStopSurplusPrice", body)

    def test_plan_order_uses_normal_plan_type(self):
        """Plan order must have planType=normal_plan."""
        body = self._build_request(self._plan_order_params())
        self.assertEqual(body.get("planType"), "normal_plan")

    def test_plan_order_has_mark_price_trigger(self):
        """Plan order trigger type should be mark_price."""
        body = self._build_request(self._plan_order_params())
        self.assertEqual(body.get("triggerType"), "mark_price")

    def test_regular_order_uses_preset_fields(self):
        """Regular (non-trigger) orders SHOULD use preset* fields."""
        body = self._build_request({
            "stopLoss": {"triggerPrice": 93000.0},
            "takeProfit": {"triggerPrice": 99000.0},
        })
        # Regular orders use presetStopLossPrice
        self.assertIn("presetStopLossPrice", body)
        self.assertIn("presetStopSurplusPrice", body)
        # And must NOT have plan-order fields
        self.assertNotIn("stopLossTriggerType", body)
        self.assertNotIn("stopSurplusTriggerType", body)


# ── Test: OrderManager full integration ──────────────────────────────


class TestPlaceTriggerOrderIntegration(unittest.TestCase):
    """Test OrderManager.place_trigger_order passes correct params through."""

    def setUp(self):
        self.mock_exchange = MagicMock()
        self.mock_exchange.create_order.return_value = {"id": "trig123"}
        self.mock_exchange.amount_to_precision.side_effect = lambda s, a: a
        self.mock_exchange.price_to_precision.side_effect = lambda s, p: p

        from config import TradingConfig
        from risk.manager import RiskManager
        from orders.manager import OrderManager

        self.exchange_client = _make_client(self.mock_exchange)

        trading_cfg = TradingConfig()
        self.risk = RiskManager(trading_cfg)
        self.order_mgr = OrderManager(self.exchange_client, self.risk)

    def _get_params(self):
        args, kwargs = self.mock_exchange.create_order.call_args
        return args[4] if len(args) > 4 else kwargs.get("params", {})

    def test_place_trigger_order_uses_mark_price(self):
        """Full flow: OrderManager -> ExchangeClient -> ccxt uses mark_price."""
        result = self.order_mgr.place_trigger_order(
            "BTC/USDT:USDT", "long", 10000.0, 95000.0,
        )
        if result is None:
            return
        params = self._get_params()
        self.assertEqual(params["triggerType"], "mark_price")

    def test_place_trigger_order_has_ccxt_sltp(self):
        """Full flow must pass ccxt unified stopLoss/takeProfit dicts."""
        result = self.order_mgr.place_trigger_order(
            "BTC/USDT:USDT", "long", 10000.0, 95000.0,
        )
        if result is None:
            return
        params = self._get_params()
        self.assertIn("stopLoss", params)
        self.assertIn("takeProfit", params)
        self.assertIsInstance(params["stopLoss"], dict)
        self.assertIsInstance(params["takeProfit"], dict)
        self.assertIn("triggerPrice", params["stopLoss"])
        self.assertIn("triggerPrice", params["takeProfit"])

    def test_place_trigger_order_result_has_sltp(self):
        """Result dict must contain SL/TP values for Telegram notifications."""
        result = self.order_mgr.place_trigger_order(
            "BTC/USDT:USDT", "long", 10000.0, 95000.0,
        )
        if result is None:
            return
        self.assertIn("stop_loss", result)
        self.assertIn("take_profit", result)
        self.assertGreater(result["stop_loss"], 0)
        self.assertGreater(result["take_profit"], 0)

    def test_place_trigger_order_no_raw_bitget_params(self):
        """Full flow must NOT produce raw Bitget params (preset*, stopSurplus*)."""
        result = self.order_mgr.place_trigger_order(
            "BTC/USDT:USDT", "long", 10000.0, 95000.0,
        )
        if result is None:
            return
        params = self._get_params()
        for bad_key in [
            "presetStopLossPrice", "presetStopLossTriggerType",
            "presetStopSurplusPrice", "presetStopSurplusTriggerType",
            "stopLossTriggerPrice", "stopLossTriggerType",
            "stopSurplusTriggerPrice", "stopSurplusTriggerType",
        ]:
            self.assertNotIn(bad_key, params,
                             f"Raw Bitget param '{bad_key}' must not be in params — use ccxt unified format")

    def test_place_trigger_order_short(self):
        """Short-side trigger order also uses ccxt unified SL/TP."""
        result = self.order_mgr.place_trigger_order(
            "ETH/USDT:USDT", "short", 10000.0, 3500.0,
        )
        if result is None:
            return
        params = self._get_params()
        self.assertIn("stopLoss", params)
        self.assertIn("takeProfit", params)
        # For short: SL > trigger price, TP < trigger price
        self.assertGreater(params["stopLoss"]["triggerPrice"], 3500.0)
        self.assertLess(params["takeProfit"]["triggerPrice"], 3500.0)


if __name__ == "__main__":
    unittest.main()
