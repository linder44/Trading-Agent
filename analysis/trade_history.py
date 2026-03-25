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

    def get_symbol_win_rate(self, symbol: str) -> float | None:
        """Get win rate for a specific symbol. Returns None if < 5 trades."""
        symbol_trades = [t for t in self._trades if t["symbol"] == symbol]
        if len(symbol_trades) < 5:
            return None
        wins = sum(1 for t in symbol_trades if t["result"] == "win")
        return wins / len(symbol_trades)

    def get_performance_analysis(self) -> dict:
        """Analyze trade performance for optimal conditions.

        Returns insights on:
        - Best/worst trading hours (UTC)
        - Optimal trade duration
        - Per-symbol edge
        - Side bias (long vs short performance)
        """
        if len(self._trades) < 10:
            return {"status": "insufficient_data", "min_trades_needed": 10}

        # Parse hours from closed_at timestamps
        hourly_stats: dict[int, dict] = {}
        duration_buckets: dict[str, dict] = {
            "0-15min": {"wins": 0, "losses": 0, "pnl": 0},
            "15-60min": {"wins": 0, "losses": 0, "pnl": 0},
            "60-120min": {"wins": 0, "losses": 0, "pnl": 0},
        }
        side_stats = {"long": {"wins": 0, "losses": 0, "pnl": 0},
                      "short": {"wins": 0, "losses": 0, "pnl": 0}}

        for t in self._trades:
            # Hourly analysis
            try:
                closed = datetime.fromisoformat(t["closed_at"])
                hour = closed.hour
            except (KeyError, ValueError):
                hour = None

            if hour is not None:
                if hour not in hourly_stats:
                    hourly_stats[hour] = {"wins": 0, "losses": 0, "pnl": 0}
                if t["result"] == "win":
                    hourly_stats[hour]["wins"] += 1
                elif t["result"] == "loss":
                    hourly_stats[hour]["losses"] += 1
                hourly_stats[hour]["pnl"] += t.get("pnl_usdt", 0)

            # Duration analysis
            dur = t.get("duration_minutes", 0)
            if dur <= 15:
                bucket = "0-15min"
            elif dur <= 60:
                bucket = "15-60min"
            else:
                bucket = "60-120min"

            if t["result"] == "win":
                duration_buckets[bucket]["wins"] += 1
            elif t["result"] == "loss":
                duration_buckets[bucket]["losses"] += 1
            duration_buckets[bucket]["pnl"] += t.get("pnl_usdt", 0)

            # Side analysis
            side = t.get("side", "long")
            if side in side_stats:
                if t["result"] == "win":
                    side_stats[side]["wins"] += 1
                elif t["result"] == "loss":
                    side_stats[side]["losses"] += 1
                side_stats[side]["pnl"] += t.get("pnl_usdt", 0)

        # Find best and worst hours
        best_hour = max(hourly_stats.items(), key=lambda x: x[1]["pnl"])[0] if hourly_stats else None
        worst_hour = min(hourly_stats.items(), key=lambda x: x[1]["pnl"])[0] if hourly_stats else None

        # Best duration bucket
        best_duration = max(duration_buckets.items(),
                            key=lambda x: x[1]["pnl"])[0] if duration_buckets else None

        # Per-symbol edge
        symbol_edge = {}
        for t in self._trades:
            sym = t["symbol"]
            if sym not in symbol_edge:
                symbol_edge[sym] = {"trades": 0, "wins": 0, "pnl": 0}
            symbol_edge[sym]["trades"] += 1
            if t["result"] == "win":
                symbol_edge[sym]["wins"] += 1
            symbol_edge[sym]["pnl"] = round(symbol_edge[sym]["pnl"] + t.get("pnl_usdt", 0), 2)

        # Calculate win rates for symbol edge
        for sym in symbol_edge:
            total = symbol_edge[sym]["trades"]
            symbol_edge[sym]["win_rate"] = round(symbol_edge[sym]["wins"] / total * 100, 1) if total > 0 else 0

        return {
            "status": "ok",
            "total_trades": len(self._trades),
            "best_hour_utc": best_hour,
            "worst_hour_utc": worst_hour,
            "best_duration": best_duration,
            "duration_stats": {k: {"pnl": round(v["pnl"], 2),
                                   "win_rate": round(v["wins"] / (v["wins"] + v["losses"]) * 100, 1)
                                   if (v["wins"] + v["losses"]) > 0 else 0}
                              for k, v in duration_buckets.items()},
            "side_stats": {k: {"pnl": round(v["pnl"], 2),
                               "win_rate": round(v["wins"] / (v["wins"] + v["losses"]) * 100, 1)
                               if (v["wins"] + v["losses"]) > 0 else 0}
                          for k, v in side_stats.items()},
            "symbol_edge": symbol_edge,
        }

    def _load(self):
        """Load from disk if exists."""
        if self.HISTORY_FILE.exists():
            try:
                self._trades = json.loads(self.HISTORY_FILE.read_text())
                logger.info(f"Загружена история: {len(self._trades)} сделок")
            except Exception as e:
                logger.warning(f"Не удалось загрузить историю: {e}")
                self._trades = []
