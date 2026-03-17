"""Trade history tracker — gives Claude memory of past decisions.

Stores completed trades with outcomes so Claude can:
- See its win rate and recent performance
- Learn from mistakes (e.g., consistently wrong on DOGE)
- Adjust confidence based on track record
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


class TradeHistoryTracker:
    """Persistent trade history with per-symbol stats."""

    MAX_HISTORY = 200  # keep last N trades
    HISTORY_FILE = Path("data/trade_history.json")

    def __init__(self):
        self._trades: list[dict] = []
        self._load()

    def record_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        amount: float,
        reason_open: str = "",
        reason_close: str = "",
        duration_minutes: float = 0,
    ):
        """Record a completed trade with its outcome."""
        if side == "long":
            pnl = (exit_price - entry_price) * amount
            pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price else 0
        else:
            pnl = (entry_price - exit_price) * amount
            pnl_pct = (entry_price - exit_price) / entry_price * 100 if entry_price else 0

        trade = {
            "symbol": symbol,
            "side": side,
            "entry_price": round(entry_price, 6),
            "exit_price": round(exit_price, 6),
            "amount": amount,
            "pnl_usdt": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "result": "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven"),
            "reason_open": reason_open,
            "reason_close": reason_close,
            "duration_minutes": round(duration_minutes, 1),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._trades.append(trade)

        # Trim to max size
        if len(self._trades) > self.MAX_HISTORY:
            self._trades = self._trades[-self.MAX_HISTORY:]

        self._save()
        logger.info(f"Trade recorded: {symbol} {side} PnL={pnl:+.2f} USDT ({pnl_pct:+.2f}%)")

    def get_summary_for_prompt(self, last_n: int = 20) -> dict:
        """Get trade history summary for Claude's context.

        Returns recent trades + aggregate stats so Claude can
        assess its own performance and adjust strategy.
        """
        if not self._trades:
            return {"total_trades": 0, "message": "Нет истории сделок"}

        recent = self._trades[-last_n:]
        all_trades = self._trades

        # Aggregate stats
        wins = [t for t in all_trades if t["result"] == "win"]
        losses = [t for t in all_trades if t["result"] == "loss"]
        total_pnl = sum(t["pnl_usdt"] for t in all_trades)

        # Per-symbol stats
        symbol_stats = {}
        for t in all_trades:
            sym = t["symbol"]
            if sym not in symbol_stats:
                symbol_stats[sym] = {"wins": 0, "losses": 0, "total_pnl": 0}
            if t["result"] == "win":
                symbol_stats[sym]["wins"] += 1
            elif t["result"] == "loss":
                symbol_stats[sym]["losses"] += 1
            symbol_stats[sym]["total_pnl"] = round(symbol_stats[sym]["total_pnl"] + t["pnl_usdt"], 2)

        # Worst performers (symbols where we lose most)
        worst = sorted(symbol_stats.items(), key=lambda x: x[1]["total_pnl"])[:3]

        win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
        avg_win = sum(t["pnl_usdt"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl_usdt"] for t in losses) / len(losses) if losses else 0

        return {
            "total_trades": len(all_trades),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl_usdt": round(total_pnl, 2),
            "avg_win_usdt": round(avg_win, 2),
            "avg_loss_usdt": round(avg_loss, 2),
            "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            "worst_symbols": [
                {"symbol": sym, **stats} for sym, stats in worst if stats["total_pnl"] < 0
            ],
            "recent_trades": [
                {
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "pnl_pct": t["pnl_pct"],
                    "result": t["result"],
                    "reason_open": t["reason_open"][:100] if t["reason_open"] else "",
                    "closed_at": t["closed_at"],
                }
                for t in recent
            ],
        }

    def _save(self):
        """Persist to disk."""
        self.HISTORY_FILE.parent.mkdir(exist_ok=True)
        self.HISTORY_FILE.write_text(json.dumps(self._trades, indent=1, ensure_ascii=False))

    def _load(self):
        """Load from disk if exists."""
        if self.HISTORY_FILE.exists():
            try:
                self._trades = json.loads(self.HISTORY_FILE.read_text())
                logger.info(f"Загружена история: {len(self._trades)} сделок")
            except Exception as e:
                logger.warning(f"Не удалось загрузить историю: {e}")
                self._trades = []
