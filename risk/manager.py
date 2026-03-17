"""Risk management module."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from config import TradingConfig


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_ids: list[str] = field(default_factory=list)


class RiskManager:
    """Manages risk, position sizing, and portfolio exposure."""

    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg
        self.positions: dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.max_daily_loss_pct: float = 0.05  # 5% max daily loss
        self.max_daily_trades: int = 20

    def calculate_position_size(self, balance: float, price: float, stop_loss_price: float) -> float:
        """Calculate position size based on risk per trade."""
        max_risk_amount = balance * self.cfg.max_position_pct
        risk_per_unit = abs(price - stop_loss_price)

        if risk_per_unit <= 0:
            logger.warning("Невалидный стоп-лосс — слишком близко к входу")
            return 0.0

        # Position size = risk amount / risk per unit
        size = max_risk_amount / risk_per_unit
        # Cap at max position value
        max_size = (balance * self.cfg.max_position_pct) / price
        size = min(size, max_size)

        return size

    def can_open_position(self, symbol: str, balance: float) -> tuple[bool, str]:
        """Check if we can open a new position."""
        if symbol in self.positions:
            return False, f"Уже есть открытая позиция по {symbol}"

        if len(self.positions) >= self.cfg.max_open_positions:
            return False, f"Достигнут максимум открытых позиций ({self.cfg.max_open_positions})"

        if self.daily_trades >= self.max_daily_trades:
            return False, f"Достигнут максимум дневных сделок ({self.max_daily_trades})"

        if self.daily_pnl < -(balance * self.max_daily_loss_pct):
            return False, f"Достигнут дневной лимит убытков ({self.max_daily_loss_pct*100}%)"

        return True, "OK"

    def compute_stop_loss(self, entry_price: float, side: str, atr: float | None = None) -> float:
        """Compute stop loss price. Uses ATR if available, otherwise fixed %."""
        if atr and atr > 0:
            # 2x ATR stop loss
            offset = atr * 2
        else:
            offset = entry_price * self.cfg.stop_loss_pct

        if side == "long":
            return entry_price - offset
        else:
            return entry_price + offset

    def compute_take_profit(self, entry_price: float, side: str, atr: float | None = None) -> float:
        """Compute take profit price. Target 2:1 risk-reward."""
        if atr and atr > 0:
            offset = atr * 4  # 2:1 RR with 2x ATR stop
        else:
            offset = entry_price * self.cfg.take_profit_pct

        if side == "long":
            return entry_price + offset
        else:
            return entry_price - offset

    def register_position(self, symbol: str, side: str, entry_price: float,
                          amount: float, stop_loss: float, take_profit: float,
                          order_ids: list[str] | None = None) -> Position:
        """Register a new open position."""
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            amount=amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_ids=order_ids or [],
        )
        self.positions[symbol] = pos
        self.daily_trades += 1
        logger.info(f"Зарегистрирована {side} позиция: {symbol} @ {entry_price}, SL={stop_loss}, TP={take_profit}")
        return pos

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close and unregister a position. Returns PnL."""
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.amount
        else:
            pnl = (pos.entry_price - exit_price) * pos.amount

        self.daily_pnl += pnl
        del self.positions[symbol]
        logger.info(f"Закрыта {pos.side} {symbol}: PnL={pnl:.2f} USDT")
        return pnl

    def get_portfolio_summary(self) -> dict:
        """Get summary of all positions for AI context."""
        return {
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "amount": p.amount,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "opened_at": p.opened_at.isoformat(),
                }
                for p in self.positions.values()
            ],
            "num_positions": len(self.positions),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
        }

    def compute_trailing_stop(self, symbol: str, current_price: float) -> float | None:
        """Compute trailing stop for an open position.

        Returns new stop loss price if it should be updated, None otherwise.
        Trailing stop only moves in the profitable direction.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        trailing_pct = self.cfg.trailing_stop_pct

        if pos.side == "long":
            # For longs, trailing stop moves up with price
            new_sl = current_price * (1 - trailing_pct)
            # Only move SL up, never down
            if new_sl > pos.stop_loss and current_price > pos.entry_price:
                return round(new_sl, 6)
        else:
            # For shorts, trailing stop moves down with price
            new_sl = current_price * (1 + trailing_pct)
            # Only move SL down, never up
            if new_sl < pos.stop_loss and current_price < pos.entry_price:
                return round(new_sl, 6)

        return None

    def check_all_trailing_stops(self, price_getter) -> list[dict]:
        """Check trailing stops for all positions.

        Args:
            price_getter: callable(symbol) -> float, returns current price

        Returns:
            List of {symbol, old_sl, new_sl} for positions that need SL update
        """
        updates = []
        for symbol, pos in self.positions.items():
            try:
                current_price = price_getter(symbol)
                new_sl = self.compute_trailing_stop(symbol, current_price)
                if new_sl is not None:
                    old_sl = pos.stop_loss
                    updates.append({
                        "symbol": symbol,
                        "old_sl": old_sl,
                        "new_sl": new_sl,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "current_price": current_price,
                    })
            except Exception as e:
                logger.warning(f"Trailing stop check failed for {symbol}: {e}")
        return updates

    def reset_daily_stats(self):
        """Reset daily counters (call at start of new day)."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        logger.info("Дневная статистика риска сброшена")
