"""Order management module - executes and tracks orders."""

from loguru import logger

from exchange.client import ExchangeClient
from risk.manager import RiskManager


class OrderManager:
    """Handles order execution, SL/TP placement, and position tracking."""

    def __init__(self, exchange: ExchangeClient, risk_mgr: RiskManager):
        self.exchange = exchange
        self.risk = risk_mgr

    def open_long(self, symbol: str, balance: float, price: float, atr: float | None = None) -> dict | None:
        """Open a long position with SL and TP."""
        can_open, reason = self.risk.can_open_position(symbol, balance)
        if not can_open:
            logger.warning(f"Cannot open long {symbol}: {reason}")
            return None

        stop_loss = self.risk.compute_stop_loss(price, "long", atr)
        take_profit = self.risk.compute_take_profit(price, "long", atr)
        amount = self.risk.calculate_position_size(balance, price, stop_loss)

        if amount <= 0:
            logger.warning(f"Position size too small for {symbol}")
            return None

        # Round amount to market precision
        market = self.exchange.get_market_info(symbol)
        amount_precision = market.get("precision", {}).get("amount", 8)
        amount = round(amount, amount_precision)

        try:
            # Place market entry
            entry_order = self.exchange.create_market_order(symbol, "buy", amount)

            # Place stop loss
            sl_order = self.exchange.create_stop_loss(
                symbol, "sell", amount, stop_loss,
                params={"reduceOnly": True}
            )

            # Place take profit
            tp_order = self.exchange.create_take_profit(
                symbol, "sell", amount, take_profit,
                params={"reduceOnly": True}
            )

            order_ids = [entry_order["id"], sl_order["id"], tp_order["id"]]
            self.risk.register_position(symbol, "long", price, amount, stop_loss, take_profit, order_ids)

            return {
                "action": "open_long",
                "symbol": symbol,
                "amount": amount,
                "entry_price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_ids": order_ids,
            }

        except Exception as e:
            logger.error(f"Failed to open long {symbol}: {e}")
            return None

    def open_short(self, symbol: str, balance: float, price: float, atr: float | None = None) -> dict | None:
        """Open a short position with SL and TP."""
        can_open, reason = self.risk.can_open_position(symbol, balance)
        if not can_open:
            logger.warning(f"Cannot open short {symbol}: {reason}")
            return None

        stop_loss = self.risk.compute_stop_loss(price, "short", atr)
        take_profit = self.risk.compute_take_profit(price, "short", atr)
        amount = self.risk.calculate_position_size(balance, price, stop_loss)

        if amount <= 0:
            return None

        market = self.exchange.get_market_info(symbol)
        amount_precision = market.get("precision", {}).get("amount", 8)
        amount = round(amount, amount_precision)

        try:
            entry_order = self.exchange.create_market_order(symbol, "sell", amount)

            sl_order = self.exchange.create_stop_loss(
                symbol, "buy", amount, stop_loss,
                params={"reduceOnly": True}
            )

            tp_order = self.exchange.create_take_profit(
                symbol, "buy", amount, take_profit,
                params={"reduceOnly": True}
            )

            order_ids = [entry_order["id"], sl_order["id"], tp_order["id"]]
            self.risk.register_position(symbol, "short", price, amount, stop_loss, take_profit, order_ids)

            return {
                "action": "open_short",
                "symbol": symbol,
                "amount": amount,
                "entry_price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_ids": order_ids,
            }

        except Exception as e:
            logger.error(f"Failed to open short {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> dict | None:
        """Close an existing position."""
        if symbol not in self.risk.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        pos = self.risk.positions[symbol]

        try:
            # Cancel existing SL/TP orders
            self.exchange.cancel_all_orders(symbol)

            # Close with market order
            close_side = "sell" if pos.side == "long" else "buy"
            close_order = self.exchange.create_market_order(
                symbol, close_side, pos.amount,
                params={"reduceOnly": True}
            )

            exit_price = float(close_order.get("average", close_order.get("price", pos.entry_price)))
            pnl = self.risk.close_position(symbol, exit_price)

            return {
                "action": "close_position",
                "symbol": symbol,
                "side": pos.side,
                "exit_price": exit_price,
                "pnl": pnl,
            }

        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return None

    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict | None:
        """Place a standalone limit order."""
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            return {
                "action": "limit_order",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "order_id": order["id"],
            }
        except Exception as e:
            logger.error(f"Failed to place limit order {symbol}: {e}")
            return None

    def update_stop_loss(self, symbol: str, new_sl_price: float) -> dict | None:
        """Update stop loss for an existing position (trailing stop)."""
        if symbol not in self.risk.positions:
            return None

        pos = self.risk.positions[symbol]

        try:
            # Cancel old SL orders (keep TP)
            open_orders = self.exchange.fetch_open_orders(symbol)
            for o in open_orders:
                if o.get("stopPrice") and o["side"] != ("sell" if pos.side == "short" else "buy"):
                    self.exchange.cancel_order(o["id"], symbol)

            # Place new SL
            sl_side = "sell" if pos.side == "long" else "buy"
            sl_order = self.exchange.create_stop_loss(
                symbol, sl_side, pos.amount, new_sl_price,
                params={"reduceOnly": True}
            )

            pos.stop_loss = new_sl_price
            logger.info(f"Updated SL for {symbol} to {new_sl_price}")

            return {"action": "update_sl", "symbol": symbol, "new_stop_loss": new_sl_price}

        except Exception as e:
            logger.error(f"Failed to update SL for {symbol}: {e}")
            return None

    def sync_positions_from_exchange(self):
        """Sync local position state with exchange."""
        try:
            exchange_positions = self.exchange.fetch_positions()
            exchange_symbols = {p["symbol"] for p in exchange_positions}

            # Remove closed positions from local tracking
            for symbol in list(self.risk.positions.keys()):
                if symbol not in exchange_symbols:
                    logger.info(f"Position {symbol} closed on exchange, removing from local tracking")
                    ticker = self.exchange.fetch_ticker(symbol)
                    self.risk.close_position(symbol, ticker["last"])

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
