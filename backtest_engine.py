"""
Rule-Based Backtesting Engine — no Claude API calls.

Uses SignalAggregator for decisions instead of Claude, making backtests:
- Free (no API costs)
- Fast (hundreds of candles per second)
- Reproducible (deterministic signals)

Simulates execution with realistic spreads, slippage, and Bitget commissions.

Usage:
    python backtest_engine.py --symbol BTC/USDT --days 30
    python backtest_engine.py --all --days 30
    python backtest_engine.py --optimize --symbol BTC/USDT --days 30
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import config
from exchange.client import ExchangeClient
from analysis.technical import TechnicalAnalyzer
from analysis.scalping import ScalpingAnalyzer
from analysis.signal_aggregator import SignalAggregator
from analysis.regime_detector import RegimeDetector
from risk.dynamic_exits import DynamicExitManager, ExitPlan

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


# Bitget fee structure
MAKER_FEE = 0.0002  # 0.02%
TAKER_FEE = 0.0006  # 0.06%
SLIPPAGE_PCT = 0.0002  # 0.02% estimated slippage


@dataclass
class BacktestPosition:
    symbol: str
    side: str
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    exit_plan: ExitPlan | None = None
    opened_at_idx: int = 0
    tp1_hit: bool = False
    tp2_hit: bool = False
    original_amount: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    days: int
    initial_balance: float
    final_balance: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self.initial_balance * 100 if self.initial_balance else 0

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> list:
        return [t for t in self.trades if t["pnl"] > 0]

    @property
    def losses(self) -> list:
        return [t for t in self.trades if t["pnl"] <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / self.num_trades * 100 if self.num_trades else 0

    @property
    def avg_win(self) -> float:
        return np.mean([t["pnl"] for t in self.wins]) if self.wins else 0

    @property
    def avg_loss(self) -> float:
        return np.mean([t["pnl"] for t in self.losses]) if self.losses else 0

    @property
    def profit_factor(self) -> float:
        total_win = sum(t["pnl"] for t in self.wins)
        total_loss = abs(sum(t["pnl"] for t in self.losses))
        return total_win / total_loss if total_loss > 0 else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.initial_balance
        max_dd = 0
        for point in self.equity_curve:
            eq = point["equity"]
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio from daily returns."""
        if len(self.equity_curve) < 2:
            return 0
        equities = [p["equity"] for p in self.equity_curve]
        returns = np.diff(equities) / equities[:-1]
        if len(returns) < 2 or np.std(returns) == 0:
            return 0
        # Assume ~480 data points per day (1m candles, 8h active trading)
        # Annualize: sqrt(365 * 480) for 1m candles
        daily_returns = []
        chunk_size = 480
        for i in range(0, len(returns), chunk_size):
            chunk = returns[i:i + chunk_size]
            daily_returns.append(np.sum(chunk))
        if len(daily_returns) < 2 or np.std(daily_returns) == 0:
            return 0
        return float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365))

    @property
    def trades_per_day(self) -> float:
        if self.days <= 0:
            return 0
        return self.num_trades / self.days


class BacktestEngine:
    """Rule-based backtesting engine — no Claude API calls."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        days: int = 30,
        initial_balance: float = 10000.0,
        # Tunable parameters
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
        trailing_pct: float = 0.003,
        min_confidence: float = 0.3,
        max_position_age: int = 90,
        max_position_pct: float = 0.08,
    ):
        self.symbol = symbol
        self.days = days
        self.initial_balance = initial_balance
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.trailing_pct = trailing_pct
        self.min_confidence = min_confidence
        self.max_position_age = max_position_age
        self.max_position_pct = max_position_pct

        self.exchange = ExchangeClient(config.bitget)
        self.analyzer = TechnicalAnalyzer()
        self.scalping = ScalpingAnalyzer()
        self.aggregator = SignalAggregator()
        self.regime_detector = RegimeDetector()
        self.exit_manager = DynamicExitManager()

        self.balance = initial_balance
        self.position: BacktestPosition | None = None
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []

    def run(self) -> BacktestResult:
        """Run the backtest."""
        logger.info(f"Backtest {self.symbol} | {self.days} days | balance={self.initial_balance}")

        # Fetch data
        swap_symbol = f"{self.symbol}:USDT" if ":USDT" not in self.symbol else self.symbol
        try:
            df_1m = self.exchange.fetch_ohlcv(swap_symbol, "1m", limit=1000)
            df_5m = self.exchange.fetch_ohlcv(swap_symbol, "5m", limit=1000)
        except Exception as e:
            logger.error(f"Failed to fetch data for {self.symbol}: {e}")
            return BacktestResult(self.symbol, self.days, self.initial_balance, self.initial_balance)

        if len(df_1m) < 200:
            logger.warning(f"Insufficient data for {self.symbol}: {len(df_1m)} candles")
            return BacktestResult(self.symbol, self.days, self.initial_balance, self.initial_balance)

        # Compute indicators on full dataset
        df_1m = self.analyzer.compute_indicators(df_1m)
        df_5m = self.analyzer.compute_indicators(df_5m)

        # Run simulation: every 3 candles (simulating 3-min cycle on 1m data)
        start_idx = 100  # Warmup period
        step = 3  # Analyze every 3 candles

        total_steps = (len(df_1m) - start_idx) // step
        logger.info(f"Running {total_steps} decision points on {len(df_1m)} candles...")

        for i in range(start_idx, len(df_1m), step):
            self._process_candle(df_1m, df_5m, i)

        # Close remaining position
        if self.position:
            last_price = float(df_1m["close"].iloc[-1])
            self._close_position(last_price, len(df_1m) - 1, "backtest_end")

        result = BacktestResult(
            symbol=self.symbol,
            days=self.days,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

        self._print_results(result)
        return result

    def _process_candle(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, idx: int):
        """Process a single decision point."""
        current_slice_1m = df_1m.iloc[:idx + 1]
        current_price = float(current_slice_1m["close"].iloc[-1])

        # Check SL/TP on intermediate candles
        if self.position:
            recent = df_1m.iloc[max(0, idx - 3):idx + 1]
            self._check_sl_tp(recent, idx)
            if not self.position:
                self._track_equity(current_price, idx)
                return

            # Time-based exit
            age = idx - self.position.opened_at_idx
            if age >= self.max_position_age:
                self._close_position(current_price, idx, "max_age")
                self._track_equity(current_price, idx)
                return

        # Generate technical summary for 1m
        summary_1m = self.analyzer.generate_summary(current_slice_1m, self.symbol)

        # Map 1m index to approximate 5m index
        ratio_5m = len(df_5m) / len(df_1m) if len(df_1m) > 0 else 0.2
        idx_5m = min(int(idx * ratio_5m), len(df_5m) - 1)
        slice_5m = df_5m.iloc[:idx_5m + 1]
        summary_5m = self.analyzer.generate_summary(slice_5m, self.symbol) if len(slice_5m) >= 50 else {}

        technical_data = {"1m": summary_1m, "5m": summary_5m}

        # Scalping analysis
        scalping_data = self.scalping.full_scalping_analysis(current_slice_1m.tail(50))

        # Regime detection
        regime = self.regime_detector.detect(slice_5m) if len(slice_5m) >= 30 else {"regime": "unknown", "strategy": "standard", "confidence_mult": 1.0}

        # Skip if choppy regime
        if regime["regime"] == "choppy":
            self._track_equity(current_price, idx)
            return

        # Get rule-based decision
        decision = self.aggregator.get_rule_based_decision(
            technical_data=technical_data,
            scalping_data=scalping_data,
            symbol=self.symbol,
        )

        # Apply regime confidence multiplier
        confidence = decision["confidence"] * regime.get("confidence_mult", 1.0)

        # Execute decision
        if confidence >= self.min_confidence and decision["action"] in ("open_long", "open_short"):
            if not self.position:
                self._open_position(decision, current_price, idx, regime)

        self._track_equity(current_price, idx)

    def _open_position(self, decision: dict, price: float, idx: int, regime: dict):
        """Open a position in backtest."""
        side = "long" if decision["action"] == "open_long" else "short"

        # Apply slippage
        if side == "long":
            entry_price = price * (1 + SLIPPAGE_PCT)
        else:
            entry_price = price * (1 - SLIPPAGE_PCT)

        # Get ATR for exit calculation
        atr = self._estimate_atr(price)

        # Dynamic exits based on regime
        exit_plan = self.exit_manager.calculate_exits(
            entry_price, side, atr, regime.get("regime", "normal"),
        )

        if exit_plan is None:
            return  # Bad R/R

        sl = exit_plan.stop_loss
        tp = exit_plan.take_profit_1  # Primary TP

        # Position sizing
        risk_per_unit = abs(entry_price - sl)
        if risk_per_unit <= 0:
            return
        max_risk = self.balance * self.max_position_pct
        amount = max_risk / risk_per_unit
        max_amount = (self.balance * self.max_position_pct) / entry_price
        amount = min(amount, max_amount)

        if amount <= 0:
            return

        # Pay entry commission
        commission = entry_price * amount * TAKER_FEE
        self.balance -= commission

        self.position = BacktestPosition(
            symbol=self.symbol,
            side=side,
            entry_price=entry_price,
            amount=amount,
            stop_loss=sl,
            take_profit=tp,
            exit_plan=exit_plan,
            opened_at_idx=idx,
            original_amount=amount,
        )

    def _close_position(self, price: float, idx: int, reason: str):
        """Close position and calculate PnL."""
        if not self.position:
            return

        # Apply slippage
        if self.position.side == "long":
            exit_price = price * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - self.position.entry_price) * self.position.amount
        else:
            exit_price = price * (1 + SLIPPAGE_PCT)
            pnl = (self.position.entry_price - exit_price) * self.position.amount

        # Exit commission
        commission = exit_price * self.position.amount * TAKER_FEE
        pnl -= commission

        self.balance += pnl
        pnl_pct = pnl / (self.position.entry_price * self.position.original_amount) * 100

        self.trades.append({
            "side": self.position.side,
            "entry": self.position.entry_price,
            "exit": exit_price,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "duration": idx - self.position.opened_at_idx,
        })
        self.position = None

    def _check_sl_tp(self, candles: pd.DataFrame, idx: int):
        """Check SL/TP/trailing stop hits."""
        if not self.position:
            return

        for _, row in candles.iterrows():
            if self.position is None:
                break

            if self.position.side == "long":
                # Check SL
                if row["low"] <= self.position.stop_loss:
                    self._close_position(self.position.stop_loss, idx, "stop_loss")
                    return
                # Check TP
                if row["high"] >= self.position.take_profit:
                    # Partial TP: close 40% at TP1
                    if self.position.exit_plan and not self.position.tp1_hit:
                        partial_amount = self.position.original_amount * 0.4
                        partial_pnl = (self.position.take_profit - self.position.entry_price) * partial_amount
                        partial_pnl -= self.position.take_profit * partial_amount * TAKER_FEE
                        self.balance += partial_pnl
                        self.position.amount -= partial_amount
                        self.position.tp1_hit = True
                        # Move SL to breakeven
                        self.position.stop_loss = self.position.entry_price
                        # Set next TP
                        self.position.take_profit = self.position.exit_plan.take_profit_2
                    else:
                        self._close_position(self.position.take_profit, idx, "take_profit")
                        return

                # Trailing stop
                if self.position.exit_plan and self.position.exit_plan.trailing_offset:
                    new_sl = row["high"] - self.position.exit_plan.trailing_offset
                    if new_sl > self.position.stop_loss and row["close"] > self.position.entry_price:
                        self.position.stop_loss = new_sl

            else:  # short
                if row["high"] >= self.position.stop_loss:
                    self._close_position(self.position.stop_loss, idx, "stop_loss")
                    return
                if row["low"] <= self.position.take_profit:
                    if self.position.exit_plan and not self.position.tp1_hit:
                        partial_amount = self.position.original_amount * 0.4
                        partial_pnl = (self.position.entry_price - self.position.take_profit) * partial_amount
                        partial_pnl -= self.position.take_profit * partial_amount * TAKER_FEE
                        self.balance += partial_pnl
                        self.position.amount -= partial_amount
                        self.position.tp1_hit = True
                        self.position.stop_loss = self.position.entry_price
                        self.position.take_profit = self.position.exit_plan.take_profit_2
                    else:
                        self._close_position(self.position.take_profit, idx, "take_profit")
                        return

                if self.position.exit_plan and self.position.exit_plan.trailing_offset:
                    new_sl = row["low"] + self.position.exit_plan.trailing_offset
                    if new_sl < self.position.stop_loss and row["close"] < self.position.entry_price:
                        self.position.stop_loss = new_sl

    def _track_equity(self, current_price: float, idx: int):
        """Track equity curve."""
        equity = self.balance
        if self.position:
            if self.position.side == "long":
                equity += (current_price - self.position.entry_price) * self.position.amount
            else:
                equity += (self.position.entry_price - current_price) * self.position.amount

        # Only track every 10 candles to save memory
        if idx % 10 == 0:
            self.equity_curve.append({"idx": idx, "equity": round(equity, 2)})

    def _estimate_atr(self, price: float) -> float:
        """Estimate ATR from price (fallback)."""
        return price * 0.003  # 0.3% of price as ATR estimate for 1m candles

    def _print_results(self, result: BacktestResult):
        """Print backtest results."""
        logger.info("=" * 60)
        logger.info(f"  BACKTEST RESULTS: {result.symbol}")
        logger.info(f"  Period: {result.days} days")
        logger.info("=" * 60)
        logger.info(f"  Initial balance: {result.initial_balance:.2f} USDT")
        logger.info(f"  Final balance:   {result.final_balance:.2f} USDT")
        logger.info(f"  Total PnL:       {result.total_pnl:+.2f} USDT ({result.total_pnl_pct:+.1f}%)")
        logger.info(f"  Total trades:    {result.num_trades}")
        logger.info(f"  Win rate:        {result.win_rate:.1f}%")
        logger.info(f"  Avg winner:      {result.avg_win:+.2f} USDT")
        logger.info(f"  Avg loser:       {result.avg_loss:+.2f} USDT")
        logger.info(f"  Profit factor:   {result.profit_factor:.2f}")
        logger.info(f"  Max drawdown:    {result.max_drawdown_pct:.1f}%")
        logger.info(f"  Sharpe ratio:    {result.sharpe_ratio:.2f}")
        logger.info(f"  Trades/day:      {result.trades_per_day:.1f}")
        logger.info("=" * 60)

        # Breakdown by reason
        if result.trades:
            reasons = {}
            for t in result.trades:
                r = t["reason"]
                if r not in reasons:
                    reasons[r] = {"count": 0, "pnl": 0}
                reasons[r]["count"] += 1
                reasons[r]["pnl"] += t["pnl"]
            logger.info("  Exit reasons:")
            for r, stats in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
                logger.info(f"    {r}: {stats['count']} trades, PnL={stats['pnl']:+.2f}")

    def save_results(self, result: BacktestResult):
        """Save backtest results to JSON."""
        Path("data").mkdir(exist_ok=True)
        data = {
            "symbol": result.symbol,
            "days": result.days,
            "initial_balance": result.initial_balance,
            "final_balance": result.final_balance,
            "total_pnl": round(result.total_pnl, 2),
            "total_pnl_pct": round(result.total_pnl_pct, 2),
            "win_rate": round(result.win_rate, 1),
            "profit_factor": round(result.profit_factor, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct, 1),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "trades_per_day": round(result.trades_per_day, 1),
            "num_trades": result.num_trades,
            "trades": result.trades,
            "params": {
                "sl_atr_mult": self.sl_atr_mult,
                "tp_atr_mult": self.tp_atr_mult,
                "trailing_pct": self.trailing_pct,
                "min_confidence": self.min_confidence,
                "max_position_age": self.max_position_age,
            },
            "run_at": datetime.now(timezone.utc).isoformat(),
        }
        filename = f"data/backtest_{result.symbol.replace('/', '_')}_{result.days}d_rule.json"
        Path(filename).write_text(json.dumps(data, indent=2))
        logger.info(f"Results saved to {filename}")


def run_parameter_sweep(symbol: str, days: int):
    """Run A/B testing of different parameter combinations."""
    logger.info("=" * 60)
    logger.info(f"  PARAMETER SWEEP: {symbol} ({days} days)")
    logger.info("=" * 60)

    param_grid = [
        {"min_confidence": 0.2, "max_position_age": 60},
        {"min_confidence": 0.3, "max_position_age": 90},
        {"min_confidence": 0.4, "max_position_age": 90},
        {"min_confidence": 0.5, "max_position_age": 90},
        {"min_confidence": 0.3, "max_position_age": 60},
        {"min_confidence": 0.3, "max_position_age": 120},
    ]

    results = []
    for params in param_grid:
        engine = BacktestEngine(symbol=symbol, days=days, **params)
        result = engine.run()
        results.append({
            "params": params,
            "pnl": result.total_pnl,
            "pnl_pct": result.total_pnl_pct,
            "win_rate": result.win_rate,
            "sharpe": result.sharpe_ratio,
            "max_dd": result.max_drawdown_pct,
            "profit_factor": result.profit_factor,
            "trades": result.num_trades,
        })

    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER SWEEP RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Params':<40} {'PnL%':>8} {'WinRate':>8} {'Sharpe':>8} {'MaxDD':>8} {'PF':>8} {'#Trades':>8}")
    logger.info("-" * 80)

    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        params_str = str(r["params"])[:38]
        logger.info(
            f"{params_str:<40} {r['pnl_pct']:>+7.1f}% {r['win_rate']:>7.1f}% "
            f"{r['sharpe']:>7.2f} {r['max_dd']:>7.1f}% {r['profit_factor']:>7.2f} {r['trades']:>7d}"
        )


def main():
    parser = argparse.ArgumentParser(description="Rule-Based Backtester")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument("--all", action="store_true", help="Backtest all symbols")
    parser.add_argument("--optimize", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    if args.optimize:
        run_parameter_sweep(args.symbol, args.days)
    elif args.all:
        for symbol in config.trading.symbols:
            engine = BacktestEngine(symbol=symbol, days=args.days)
            result = engine.run()
            engine.save_results(result)
    else:
        engine = BacktestEngine(symbol=args.symbol, days=args.days)
        result = engine.run()
        engine.save_results(result)


if __name__ == "__main__":
    main()
