"""
Backtesting module for the AI Trading Agent.

Tests how the agent would have traded on historical data.
Note: Uses Claude API calls, so each backtest candle costs money.
For cost efficiency, it analyzes key points rather than every candle.

Usage:
    python backtest.py --symbol BTC/USDT --days 30
    python backtest.py --symbol ETH/USDT --days 7 --interval 4h
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from config import config
from exchange.client import ExchangeClient
from analysis.technical import TechnicalAnalyzer
from analysis.patterns import PatternRecognizer
from risk.manager import RiskManager
from agent.brain import TradingBrain

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


class Backtester:
    """Simulates trading on historical data using Claude AI decisions."""

    def __init__(self, symbol: str, days: int, interval: str = "4h"):
        self.symbol = symbol
        self.days = days
        self.interval = interval

        self.exchange = ExchangeClient(config.bitget)
        self.analyzer = TechnicalAnalyzer()
        self.patterns = PatternRecognizer()
        self.risk = RiskManager(config.trading)
        self.brain = TradingBrain(config.claude)

        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = None  # {"side", "entry_price", "amount", "sl", "tp"}
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []

    def run(self):
        """Run the backtest."""
        logger.info(f"Backtesting {self.symbol} | {self.days} days | {self.interval} interval")
        logger.info(f"Initial balance: {self.initial_balance} USDT")

        # Fetch historical data
        candles_needed = self.days * (24 // self._interval_hours()) + 200  # +200 for indicator warmup
        candles_needed = min(candles_needed, 1000)

        logger.info(f"Fetching {candles_needed} candles...")
        df = self.exchange.fetch_ohlcv(self.symbol, self.interval, limit=candles_needed)

        # Compute indicators on full dataset
        df = self.analyzer.compute_indicators(df)

        # Skip first 200 candles (indicator warmup)
        start_idx = 200
        analysis_points = range(start_idx, len(df), 6)  # Every 6 candles to save API costs

        logger.info(f"Analyzing {len(list(analysis_points))} decision points...")
        analysis_points = range(start_idx, len(df), 6)  # Re-create generator

        for i in analysis_points:
            current_slice = df.iloc[:i+1]
            current_price = float(current_slice["close"].iloc[-1])
            current_time = current_slice.index[-1]

            # Check SL/TP hits
            if self.position:
                self._check_sl_tp(df.iloc[max(0, i-5):i+1])

            # Generate technical summary
            summary = self.analyzer.generate_summary(current_slice, self.symbol)
            pattern_analysis = self.patterns.get_full_pattern_analysis(current_slice)

            # Build simplified portfolio
            portfolio = {
                "open_positions": [],
                "num_positions": 0,
                "daily_pnl": 0,
                "daily_trades": len(self.trades),
            }
            if self.position:
                unrealized_pnl = self._calc_unrealized_pnl(current_price)
                portfolio["open_positions"] = [{
                    "symbol": self.symbol,
                    "side": self.position["side"],
                    "entry_price": self.position["entry_price"],
                    "unrealized_pnl": round(unrealized_pnl, 2),
                }]
                portfolio["num_positions"] = 1

            # Ask Claude (simplified — no news/social for backtest speed)
            technical_data = {self.symbol: {self.interval: summary}}
            pattern_data = {self.symbol: {self.interval: pattern_analysis}}

            try:
                decision = self.brain.analyze_and_decide(
                    technical_data=technical_data,
                    market_context={"note": "Backtest mode - no live news"},
                    portfolio=portfolio,
                    balance=self.balance,
                    pattern_data=pattern_data,
                )

                for d in decision.get("decisions", []):
                    if d.get("confidence", 0) >= 0.6:
                        self._execute_backtest_decision(d, current_price, current_time)

            except Exception as e:
                logger.warning(f"Decision failed at {current_time}: {e}")

            # Track equity
            equity = self.balance
            if self.position:
                equity += self._calc_unrealized_pnl(current_price)
            self.equity_curve.append({
                "time": str(current_time),
                "equity": round(equity, 2),
                "balance": round(self.balance, 2),
            })

            # Rate limit for Claude API
            time.sleep(1)

        # Close any remaining position at last price
        if self.position:
            last_price = float(df["close"].iloc[-1])
            self._close_position(last_price, df.index[-1], "backtest_end")

        self._print_results()
        self._save_results()

    def _execute_backtest_decision(self, decision: dict, price: float, timestamp):
        """Execute a decision in backtest mode."""
        action = decision["action"]
        symbol = decision.get("symbol", self.symbol)

        if action == "open_long" and not self.position:
            atr = self._get_atr_from_price(price)
            sl = self.risk.compute_stop_loss(price, "long", atr)
            tp = self.risk.compute_take_profit(price, "long", atr)
            amount = self.risk.calculate_position_size(self.balance, price, sl)

            if amount > 0:
                self.position = {
                    "side": "long", "entry_price": price,
                    "amount": amount, "sl": sl, "tp": tp,
                }
                logger.info(f"  [{timestamp}] LONG {symbol} @ {price} | SL: {sl:.2f} TP: {tp:.2f}")

        elif action == "open_short" and not self.position:
            atr = self._get_atr_from_price(price)
            sl = self.risk.compute_stop_loss(price, "short", atr)
            tp = self.risk.compute_take_profit(price, "short", atr)
            amount = self.risk.calculate_position_size(self.balance, price, sl)

            if amount > 0:
                self.position = {
                    "side": "short", "entry_price": price,
                    "amount": amount, "sl": sl, "tp": tp,
                }
                logger.info(f"  [{timestamp}] SHORT {symbol} @ {price} | SL: {sl:.2f} TP: {tp:.2f}")

        elif action == "close" and self.position:
            self._close_position(price, timestamp, decision.get("reason", ""))

    def _close_position(self, price: float, timestamp, reason: str):
        """Close position and calculate PnL."""
        if not self.position:
            return

        if self.position["side"] == "long":
            pnl = (price - self.position["entry_price"]) * self.position["amount"]
        else:
            pnl = (self.position["entry_price"] - price) * self.position["amount"]

        self.balance += pnl
        self.trades.append({
            "time": str(timestamp),
            "side": self.position["side"],
            "entry": self.position["entry_price"],
            "exit": price,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (self.position["entry_price"] * self.position["amount"]) * 100, 2),
            "reason": reason,
        })
        logger.info(f"  [{timestamp}] CLOSE @ {price} | PnL: {pnl:+.2f} USDT ({self.trades[-1]['pnl_pct']:+.1f}%)")
        self.position = None

    def _check_sl_tp(self, candles):
        """Check if stop loss or take profit was hit in recent candles."""
        if not self.position:
            return

        for idx, row in candles.iterrows():
            if self.position["side"] == "long":
                if row["low"] <= self.position["sl"]:
                    self._close_position(self.position["sl"], idx, "stop_loss_hit")
                    return
                if row["high"] >= self.position["tp"]:
                    self._close_position(self.position["tp"], idx, "take_profit_hit")
                    return
            else:  # short
                if row["high"] >= self.position["sl"]:
                    self._close_position(self.position["sl"], idx, "stop_loss_hit")
                    return
                if row["low"] <= self.position["tp"]:
                    self._close_position(self.position["tp"], idx, "take_profit_hit")
                    return

    def _calc_unrealized_pnl(self, current_price: float) -> float:
        if not self.position:
            return 0
        if self.position["side"] == "long":
            return (current_price - self.position["entry_price"]) * self.position["amount"]
        return (self.position["entry_price"] - current_price) * self.position["amount"]

    def _get_atr_from_price(self, price: float) -> float:
        """Estimate ATR as 2% of price if not available."""
        return price * 0.02

    def _interval_hours(self) -> int:
        mapping = {"1h": 1, "4h": 4, "1d": 24, "15m": 1}
        return mapping.get(self.interval, 4)

    def _print_results(self):
        """Print backtest results."""
        total_trades = len(self.trades)
        if total_trades == 0:
            logger.info("No trades executed during backtest.")
            return

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in self.trades)
        win_rate = len(wins) / total_trades * 100

        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else float('inf')

        max_drawdown = 0
        peak = self.initial_balance
        for point in self.equity_curve:
            peak = max(peak, point["equity"])
            dd = (peak - point["equity"]) / peak * 100
            max_drawdown = max(max_drawdown, dd)

        logger.info("=" * 50)
        logger.info(f"  BACKTEST RESULTS: {self.symbol}")
        logger.info(f"  Period: {self.days} days | Interval: {self.interval}")
        logger.info("=" * 50)
        logger.info(f"  Initial Balance:  {self.initial_balance:.2f} USDT")
        logger.info(f"  Final Balance:    {self.balance:.2f} USDT")
        logger.info(f"  Total PnL:        {total_pnl:+.2f} USDT ({total_pnl/self.initial_balance*100:+.1f}%)")
        logger.info(f"  Total Trades:     {total_trades}")
        logger.info(f"  Win Rate:         {win_rate:.1f}%")
        logger.info(f"  Avg Win:          {avg_win:+.2f} USDT")
        logger.info(f"  Avg Loss:         {avg_loss:+.2f} USDT")
        logger.info(f"  Profit Factor:    {profit_factor:.2f}")
        logger.info(f"  Max Drawdown:     {max_drawdown:.1f}%")
        logger.info("=" * 50)

    def _save_results(self):
        """Save backtest results to file."""
        Path("data").mkdir(exist_ok=True)
        results = {
            "symbol": self.symbol,
            "days": self.days,
            "interval": self.interval,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_pnl": round(self.balance - self.initial_balance, 2),
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "run_at": datetime.utcnow().isoformat(),
        }
        filename = f"data/backtest_{self.symbol.replace('/', '_')}_{self.days}d.json"
        Path(filename).write_text(json.dumps(results, indent=2))
        logger.info(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="AI Trading Agent Backtester")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--interval", default="4h", help="Candle interval (1h, 4h, 1d)")
    args = parser.parse_args()

    backtester = Backtester(args.symbol, args.days, args.interval)
    backtester.run()


if __name__ == "__main__":
    main()
