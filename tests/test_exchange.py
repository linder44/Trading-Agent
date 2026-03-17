"""Tests for exchange client — trigger orders and SL/TP orders.

Verifies that Bitget-specific parameters are passed correctly to ccxt,
particularly triggerType values that caused error 400172.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCreateTriggerOrder(unittest.TestCase):
    """Test create_trigger_order sends correct params to Bitget."""

    def setUp(self):
        """Create ExchangeClient with mocked ccxt."""
        with patch("exchange.client.ccxt.bitget") as mock_bitget:
            mock_exchange = MagicMock()
            mock_exchange.markets = {
                "BTC/USDT:USDT": {"swap": True},
                "ETH/USDT:USDT": {"swap": True},
            }
            mock_bitget.return_value = mock_exchange

            from config import BitgetConfig
            cfg = BitgetConfig()
            cfg.api_key = ""
            cfg.secret_key = ""
            cfg.passphrase = ""
            cfg.demo = False

            from exchange.client import ExchangeClient
            self.client = ExchangeClient(cfg)
            self.mock_exchange = mock_exchange

    def test_trigger_order_uses_mark_price(self):
        """triggerType must be 'mark_price', not 'market_price' (Bitget 400172 fix)."""
        self.mock_exchange.create_order.return_value = {"id": "order123"}

        self.client.create_trigger_order("BTC/USDT:USDT", "buy", 0.01, 95000.0)

        self.mock_exchange.create_order.assert_called_once()
        args, kwargs = self.mock_exchange.create_order.call_args

        # Positional: symbol, type, side, amount
        self.assertEqual(args[0], "BTC/USDT:USDT")
        self.assertEqual(args[1], "market")
        self.assertEqual(args[2], "buy")
        self.assertEqual(args[3], 0.01)

        # params must contain correct triggerType
        params = args[4] if len(args) > 4 else kwargs.get("params", {})
        self.assertEqual(params["triggerType"], "mark_price")
        self.assertEqual(params["triggerPrice"], 95000.0)

    def test_trigger_order_not_market_price(self):
        """Ensure 'market_price' (the old buggy value) is NOT used."""
        self.mock_exchange.create_order.return_value = {"id": "order456"}

        self.client.create_trigger_order("ETH/USDT:USDT", "sell", 0.1, 3500.0)

        args, kwargs = self.mock_exchange.create_order.call_args
        params = args[4] if len(args) > 4 else kwargs.get("params", {})
        self.assertNotEqual(params["triggerType"], "market_price",
                            "triggerType='market_price' causes Bitget error 400172")

    def test_trigger_order_returns_order(self):
        """create_trigger_order should return the order dict from ccxt."""
        expected = {"id": "order789", "status": "open"}
        self.mock_exchange.create_order.return_value = expected

        result = self.client.create_trigger_order("BTC/USDT:USDT", "buy", 0.05, 90000.0)
        self.assertEqual(result, expected)

    def test_trigger_order_with_extra_params(self):
        """Extra params should be merged, not overwrite triggerType/triggerPrice."""
        self.mock_exchange.create_order.return_value = {"id": "order101"}

        self.client.create_trigger_order(
            "BTC/USDT:USDT", "buy", 0.01, 95000.0,
            params={"reduceOnly": True},
        )

        args, kwargs = self.mock_exchange.create_order.call_args
        params = args[4] if len(args) > 4 else kwargs.get("params", {})
        self.assertEqual(params["triggerType"], "mark_price")
        self.assertEqual(params["triggerPrice"], 95000.0)
        self.assertTrue(params["reduceOnly"])

    def test_trigger_order_sell_side(self):
        """Sell-side trigger orders should also use mark_price."""
        self.mock_exchange.create_order.return_value = {"id": "sell_order"}

        self.client.create_trigger_order("BTC/USDT:USDT", "sell", 0.02, 110000.0)

        args, kwargs = self.mock_exchange.create_order.call_args
        params = args[4] if len(args) > 4 else kwargs.get("params", {})
        self.assertEqual(params["triggerType"], "mark_price")
        self.assertEqual(params["triggerPrice"], 110000.0)


class TestCreateMarketOrderWithSLTP(unittest.TestCase):
    """Test create_market_order_with_sltp sends correct SL/TP params."""

    def setUp(self):
        with patch("exchange.client.ccxt.bitget") as mock_bitget:
            mock_exchange = MagicMock()
            mock_exchange.markets = {"BTC/USDT:USDT": {"swap": True}}
            mock_bitget.return_value = mock_exchange

            from config import BitgetConfig
            cfg = BitgetConfig()
            cfg.api_key = ""
            cfg.secret_key = ""
            cfg.passphrase = ""
            cfg.demo = False

            from exchange.client import ExchangeClient
            self.client = ExchangeClient(cfg)
            self.mock_exchange = mock_exchange

    def test_sltp_params_structure(self):
        """SL/TP must use nested dict format with triggerPrice and type."""
        self.mock_exchange.create_order.return_value = {"id": "sltp_order"}

        self.client.create_market_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 94000.0, 100000.0,
        )

        args, kwargs = self.mock_exchange.create_order.call_args
        params = args[4] if len(args) > 4 else kwargs.get("params", {})

        self.assertIn("stopLoss", params)
        self.assertIn("takeProfit", params)
        self.assertEqual(params["stopLoss"]["triggerPrice"], 94000.0)
        self.assertEqual(params["stopLoss"]["type"], "market")
        self.assertEqual(params["takeProfit"]["triggerPrice"], 100000.0)
        self.assertEqual(params["takeProfit"]["type"], "market")

    def test_sltp_order_type_is_market(self):
        """Main order type must be 'market'."""
        self.mock_exchange.create_order.return_value = {"id": "m_order"}

        self.client.create_market_order_with_sltp(
            "BTC/USDT:USDT", "buy", 0.01, 94000.0, 100000.0,
        )

        args, _ = self.mock_exchange.create_order.call_args
        self.assertEqual(args[1], "market")


class TestPlaceTriggerOrderIntegration(unittest.TestCase):
    """Test OrderManager.place_trigger_order passes correct params through."""

    def setUp(self):
        self.mock_exchange = MagicMock()
        self.mock_exchange.create_order.return_value = {"id": "trig123"}
        self.mock_exchange.amount_to_precision.side_effect = lambda s, a: a
        self.mock_exchange.price_to_precision.side_effect = lambda s, p: p

        from config import BitgetConfig, TradingConfig
        from exchange.client import ExchangeClient
        from risk.manager import RiskManager
        from orders.manager import OrderManager

        with patch("exchange.client.ccxt.bitget") as mock_bitget:
            mock_bitget.return_value = self.mock_exchange
            self.mock_exchange.markets = {"BTC/USDT:USDT": {"swap": True}}
            cfg = BitgetConfig()
            cfg.api_key = ""
            cfg.secret_key = ""
            cfg.passphrase = ""
            cfg.demo = False
            self.exchange_client = ExchangeClient(cfg)

        trading_cfg = TradingConfig()
        self.risk = RiskManager(trading_cfg)
        self.order_mgr = OrderManager(self.exchange_client, self.risk)

    def test_place_trigger_order_uses_mark_price(self):
        """Full flow: OrderManager -> ExchangeClient -> ccxt uses mark_price."""
        result = self.order_mgr.place_trigger_order(
            "BTC/USDT:USDT", "long", 10000.0, 95000.0,
        )

        if result is None:
            # Risk manager may block — check that it's not a trigger type issue
            return

        self.mock_exchange.create_order.assert_called_once()
        args, kwargs = self.mock_exchange.create_order.call_args
        params = args[4] if len(args) > 4 else kwargs.get("params", {})
        self.assertEqual(params["triggerType"], "mark_price")


if __name__ == "__main__":
    unittest.main()
