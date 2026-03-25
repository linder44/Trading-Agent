"""Drawdown Circuit Breaker — stepped loss protection.

Instead of a single 5% daily stop, implements graduated response:
- -1% daily PnL → reduce position sizes 50%
- -2% daily PnL → only trade Tier 1 signals (highest confidence)
- -3% daily PnL → stop trading for 2 hours (cooldown)
- -4% daily PnL → stop trading until next day

Also tracks consecutive losses for streak-based protection.
"""

import time
from datetime import datetime, timezone
from loguru import logger


class DrawdownBreaker:
    """Graduated drawdown protection system."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.daily_pnl: float = 0.0
        self.daily_pnl_pct: float = 0.0
        self.consecutive_losses: int = 0
        self.cooldown_until: float = 0  # Unix timestamp
        self.stopped_for_day: bool = False
        self._last_reset_date: str = ""

    def update(self, pnl: float, balance: float):
        """Update after a trade closes."""
        self.daily_pnl += pnl
        self.daily_pnl_pct = self.daily_pnl / balance * 100 if balance > 0 else 0

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check if we need to stop for the day
        if self.daily_pnl_pct <= -4.0:
            self.stopped_for_day = True
            logger.warning("DRAWDOWN BREAKER: -4% daily loss — STOPPED for the day")

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed.

        Returns (allowed, reason).
        """
        # Daily reset
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self.reset_daily()
            self._last_reset_date = today

        if self.stopped_for_day:
            return False, "Daily loss limit (-4%) reached — stopped until tomorrow"

        if time.time() < self.cooldown_until:
            remaining = int(self.cooldown_until - time.time())
            return False, f"Cooldown active — {remaining}s remaining after -3% drawdown"

        if self.daily_pnl_pct <= -3.0:
            self.cooldown_until = time.time() + 7200  # 2 hour cooldown
            return False, "3% daily loss — 2 hour cooldown activated"

        return True, "OK"

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on drawdown state.

        1.0 = normal, 0.5 = reduced, 0.0 = no trading.
        """
        if self.stopped_for_day:
            return 0.0

        # After 3 consecutive losses → 50% size
        if self.consecutive_losses >= 3:
            return 0.5

        # -2% daily → 50% size
        if self.daily_pnl_pct <= -2.0:
            return 0.5

        # -1% daily → 75% size
        if self.daily_pnl_pct <= -1.0:
            return 0.75

        return 1.0

    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold based on drawdown state.

        Higher threshold when losing = more selective.
        """
        if self.consecutive_losses >= 3:
            return 0.8  # Only highest confidence trades

        if self.daily_pnl_pct <= -2.0:
            return 0.8  # Only Tier 1 signals

        if self.daily_pnl_pct <= -1.0:
            return 0.7

        return 0.6  # Normal threshold

    def get_status(self) -> dict:
        """Get current drawdown breaker status."""
        return {
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "consecutive_losses": self.consecutive_losses,
            "position_size_mult": self.get_position_size_multiplier(),
            "min_confidence": self.get_min_confidence(),
            "stopped_for_day": self.stopped_for_day,
            "in_cooldown": time.time() < self.cooldown_until,
        }

    def reset_daily(self):
        """Reset all daily counters."""
        self.daily_pnl = 0.0
        self.daily_pnl_pct = 0.0
        self.consecutive_losses = 0
        self.cooldown_until = 0
        self.stopped_for_day = False
        logger.info("Drawdown breaker: daily stats reset")
