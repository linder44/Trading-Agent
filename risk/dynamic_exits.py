"""Dynamic Exit Manager — context-aware SL/TP and partial take profits.

Instead of fixed SL=0.5%, TP=1.0%, exits adapt to:
- Market regime (trending → wide TP, ranging → tight TP)
- ATR (volatility-based distances)
- Structure (nearest S/R levels)

Also implements:
- Partial take profit (40%/30%/30% cascade)
- Time-based exit rules
- Breakeven stop after TP1
"""

from dataclasses import dataclass
from loguru import logger


@dataclass
class ExitPlan:
    """Complete exit plan for a position."""
    stop_loss: float
    take_profit_1: float  # 40% of position
    take_profit_2: float  # 30% of position
    take_profit_3: float  # 30% trailing
    trailing_offset: float | None  # ATR-based trailing distance
    risk_reward: float
    regime: str


class DynamicExitManager:
    """Computes context-aware exit levels."""

    def calculate_exits(
        self,
        entry_price: float,
        direction: str,  # "long" or "short"
        atr: float | None,
        regime: str = "normal",
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
    ) -> ExitPlan | None:
        """Calculate dynamic exit levels based on market context.

        Returns None if risk/reward is unacceptable (< 1.5).
        """
        if not atr or atr <= 0:
            atr = entry_price * 0.005  # Fallback: 0.5% of price

        if regime == "strong_trend":
            # Trending: tight SL, wide TP — let profits run
            sl_distance = 1.0 * atr
            tp1_distance = 1.5 * atr
            tp2_distance = 3.0 * atr
            tp3_distance = 5.0 * atr
            trailing = 0.4 * atr
        elif regime == "range" or regime == "mean_reversion":
            # Ranging: SL near structure, TP at opposite boundary
            if direction == "long" and nearest_support and nearest_resistance:
                sl_distance = max(entry_price - nearest_support + 0.1 * atr, 0.5 * atr)
                tp_total = max(nearest_resistance - entry_price - 0.1 * atr, 1.0 * atr)
                tp1_distance = tp_total * 0.5
                tp2_distance = tp_total * 0.8
                tp3_distance = tp_total
            elif direction == "short" and nearest_support and nearest_resistance:
                sl_distance = max(nearest_resistance - entry_price + 0.1 * atr, 0.5 * atr)
                tp_total = max(entry_price - nearest_support - 0.1 * atr, 1.0 * atr)
                tp1_distance = tp_total * 0.5
                tp2_distance = tp_total * 0.8
                tp3_distance = tp_total
            else:
                sl_distance = 1.0 * atr
                tp1_distance = 1.0 * atr
                tp2_distance = 1.5 * atr
                tp3_distance = 2.0 * atr
            trailing = None  # No trailing in range (price oscillates)
        elif regime in ("expanding", "volatile"):
            # High volatility: wider SL, quick TP — grab and run
            sl_distance = 2.0 * atr
            tp1_distance = 1.5 * atr
            tp2_distance = 2.5 * atr
            tp3_distance = 3.0 * atr
            trailing = 0.6 * atr
        elif regime == "squeeze" or regime == "breakout":
            # Breakout: moderate SL, wide TP
            sl_distance = 1.5 * atr
            tp1_distance = 2.0 * atr
            tp2_distance = 4.0 * atr
            tp3_distance = 6.0 * atr
            trailing = 0.5 * atr
        else:
            # Normal/default
            sl_distance = 1.5 * atr
            tp1_distance = 1.5 * atr
            tp2_distance = 3.0 * atr
            tp3_distance = 4.5 * atr
            trailing = 0.4 * atr

        # Compute actual prices
        if direction == "long":
            sl = entry_price - sl_distance
            tp1 = entry_price + tp1_distance
            tp2 = entry_price + tp2_distance
            tp3 = entry_price + tp3_distance
        else:
            sl = entry_price + sl_distance
            tp1 = entry_price - tp1_distance
            tp2 = entry_price - tp2_distance
            tp3 = entry_price - tp3_distance

        # Check risk/reward
        rr = tp1_distance / sl_distance if sl_distance > 0 else 0
        if rr < 1.2:
            # Try with TP2 as the reference
            rr = tp2_distance / sl_distance if sl_distance > 0 else 0
            if rr < 1.5:
                logger.debug(f"Bad R/R ({rr:.2f}) for {direction} @ {entry_price}, regime={regime}")
                return None

        return ExitPlan(
            stop_loss=round(sl, 6),
            take_profit_1=round(tp1, 6),
            take_profit_2=round(tp2, 6),
            take_profit_3=round(tp3, 6),
            trailing_offset=round(trailing, 6) if trailing else None,
            risk_reward=round(rr, 2),
            regime=regime,
        )

    @staticmethod
    def should_move_to_breakeven(
        entry_price: float,
        current_price: float,
        direction: str,
        atr: float,
        commission_pct: float = 0.06,
    ) -> bool:
        """Check if SL should be moved to breakeven.

        Move to breakeven when profit > 0.3% or > 1.5x ATR.
        """
        if direction == "long":
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100

        if profit_pct > 0.3:
            return True
        if atr > 0 and abs(current_price - entry_price) > 1.5 * atr:
            return True
        return False

    @staticmethod
    def time_based_exit_check(
        age_minutes: float,
        current_pnl_pct: float,
    ) -> str | None:
        """Check time-based exit rules.

        Returns:
          "close" — close immediately
          "tighten" — move SL to breakeven
          None — no action
        """
        # Position in profit > 0.3% after 30 min → move to breakeven
        if age_minutes >= 30 and current_pnl_pct > 0.3:
            return "tighten"

        # Position barely profitable after 60 min → close (dead trade)
        if age_minutes >= 60 and current_pnl_pct < 0.1:
            return "close"

        # Position in loss after 45 min with no improvement → close
        if age_minutes >= 45 and current_pnl_pct < 0:
            return "close"

        # Maximum position age: 90 min (not 120)
        if age_minutes >= 90:
            return "close"

        return None
