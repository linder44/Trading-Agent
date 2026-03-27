"""MACD + 200 EMA + S/R Strategy configuration."""

STRATEGY_CONFIG = {
    # ── MACD Parameters ──
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # ── Trend Filter ──
    "ema_period": 200,

    # ── S/R Confirmation ──
    "sr_proximity_pct": 0.5,       # price within 0.5% of S/R level = confirmed
    "sr_lookbacks": (20, 50, 100),  # swing detection windows

    # ── Risk Management ──
    "rr_ratio": 1.5,               # TP = 1.5x SL distance
    "atr_sl_buffer": 0.5,          # ATR multiplier for SL buffer beyond EMA 200
    "max_risk_pct": 2.0,           # max % of portfolio risk per trade
    "base_position_pct": 5.0,      # % of portfolio per trade
    "max_position_pct": 8.0,       # max per trade
    "max_positions": 5,

    # ── Entry Filters ──
    "min_confidence": 0.6,
    "min_atr_pct": 0.05,           # skip if ATR/price < 0.05% (dead market)

    # ── Position Management ──
    "dead_trade_minutes": 60,      # close if flat
    "max_trade_minutes": 120,      # absolute max
    "breakeven_pnl_pct": 0.3,     # move SL to BE at this PnL %

    # ── Risk Throttle ──
    "daily_loss_reduced": -1.5,
    "daily_loss_critical": -3.0,
    "daily_loss_stop": -4.0,
    "max_consecutive_losses": 3,
    "symbol_cooldown_minutes": 30,

    # ── Cycle Timing ──
    "signal_timeframe": "5m",      # MACD crossover detection
    "trend_timeframe": "15m",      # EMA 200 trend confirmation
    "max_daily_trades": 20,
}
