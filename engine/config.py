"""Signal Engine configuration — all tunable parameters in one place."""

ENGINE_CONFIG = {
    # ── Entry thresholds ──
    "min_score": 3.0,             # |score| < 3.0 = no entry
    "min_confidence": 0.4,        # confidence < 0.4 = no entry
    "strong_signal_score": 7.0,   # |score| > 7.0 = increased size

    # ── Gate keeper ──
    "max_spread_pct": 0.15,       # max spread for entry
    "min_volume_ratio": 0.5,      # min RVOL
    "max_positions": 5,
    "symbol_cooldown_minutes": 30, # pause after 2 losses on same symbol

    # ── Position sizing ──
    "base_position_pct": 5.0,     # % of portfolio per trade
    "max_position_pct": 8.0,      # max per trade
    "max_daily_trades": 20,

    # ── Time exits ──
    "dead_trade_minutes": 45,     # close if flat
    "losing_trade_minutes": 30,   # close if losing + trend against
    "max_trade_minutes": 90,      # absolute max
    "breakeven_pnl_pct": 0.3,    # move SL to BE at this PnL %

    # ── Risk throttle ──
    "daily_loss_reduced": -1.5,   # reduce size
    "daily_loss_critical": -3.0,  # only strong signals
    "daily_loss_stop": -4.0,      # stop trading
    "max_consecutive_losses": 3,  # loss streak for reduced

    # ── Cycle timing ──
    "fast_cycle_seconds": 30,
    "full_cycle_seconds": 120,

    # ── Tier weights ──
    "tier1_weight": 3.0,
    "tier2_weight": 2.0,
    "tier3_weight": 1.0,
}
