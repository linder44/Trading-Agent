"""Tests for scalping-specific risk management (position expiry, tighter SL/TP)."""

import pytest
from datetime import datetime, timedelta, timezone

from config import TradingConfig
from risk.manager import RiskManager, Position


@pytest.fixture
def risk_mgr():
    cfg = TradingConfig()
    return RiskManager(cfg)


def test_config_scalping_values():
    """Verify config has scalping-appropriate values."""
    cfg = TradingConfig()
    assert cfg.stop_loss_pct == 0.005  # 0.5%
    assert cfg.take_profit_pct == 0.01  # 1.0%
    assert cfg.trailing_stop_pct == 0.003  # 0.3%
    assert cfg.analysis_interval_minutes == 3
    assert cfg.max_position_age_minutes == 120
    assert cfg.max_open_positions == 5
    assert "1m" in cfg.timeframes
    assert "5m" in cfg.timeframes
    assert "15m" in cfg.timeframes


def test_get_expired_positions_none(risk_mgr):
    """No positions = no expired."""
    expired = risk_mgr.get_expired_positions(120)
    assert expired == []


def test_get_expired_positions_fresh(risk_mgr):
    """Fresh position should not be expired."""
    risk_mgr.register_position("BTC/USDT", "long", 50000, 0.1, 49750, 50500)
    expired = risk_mgr.get_expired_positions(120)
    assert expired == []


def test_get_expired_positions_old(risk_mgr):
    """Position older than max_age should be expired."""
    risk_mgr.register_position("BTC/USDT", "long", 50000, 0.1, 49750, 50500)
    # Manually set opened_at to 3 hours ago
    risk_mgr.positions["BTC/USDT"].opened_at = datetime.now(timezone.utc) - timedelta(hours=3)
    expired = risk_mgr.get_expired_positions(120)
    assert "BTC/USDT" in expired


def test_get_expired_positions_mixed(risk_mgr):
    """Mix of fresh and old positions."""
    risk_mgr.register_position("BTC/USDT", "long", 50000, 0.1, 49750, 50500)
    risk_mgr.register_position("ETH/USDT", "short", 3000, 1, 3015, 2970)
    # Make BTC old
    risk_mgr.positions["BTC/USDT"].opened_at = datetime.now(timezone.utc) - timedelta(minutes=150)
    expired = risk_mgr.get_expired_positions(120)
    assert "BTC/USDT" in expired
    assert "ETH/USDT" not in expired


def test_get_position_age_minutes(risk_mgr):
    """Test position age calculation."""
    risk_mgr.register_position("BTC/USDT", "long", 50000, 0.1, 49750, 50500)
    age = risk_mgr.get_position_age_minutes("BTC/USDT")
    assert age is not None
    assert age < 1  # Just created, should be < 1 minute


def test_get_position_age_no_position(risk_mgr):
    """No position returns None."""
    age = risk_mgr.get_position_age_minutes("BTC/USDT")
    assert age is None


def test_portfolio_summary_has_age(risk_mgr):
    """Portfolio summary should include age_minutes field."""
    risk_mgr.register_position("BTC/USDT", "long", 50000, 0.1, 49750, 50500)
    summary = risk_mgr.get_portfolio_summary()
    positions = summary["open_positions"]
    assert len(positions) == 1
    assert "age_minutes" in positions[0]
    assert positions[0]["age_minutes"] >= 0


def test_scalping_stop_loss_tighter(risk_mgr):
    """SL should use 1.5x ATR (not 2x) for scalping."""
    atr = 100
    sl = risk_mgr.compute_stop_loss(50000, "long", atr)
    # 1.5x ATR = 150, so SL = 50000 - 150 = 49850
    assert sl == 50000 - 150


def test_scalping_take_profit(risk_mgr):
    """TP should use 3x ATR (2:1 RR with 1.5x ATR stop)."""
    atr = 100
    tp = risk_mgr.compute_take_profit(50000, "long", atr)
    # 3x ATR = 300, so TP = 50000 + 300 = 50300
    assert tp == 50000 + 300


def test_scalping_fixed_sl(risk_mgr):
    """Without ATR, use fixed 0.5% SL."""
    sl = risk_mgr.compute_stop_loss(50000, "long")
    expected = 50000 - 50000 * 0.005  # 49750
    assert sl == expected


def test_scalping_fixed_tp(risk_mgr):
    """Without ATR, use fixed 1.0% TP."""
    tp = risk_mgr.compute_take_profit(50000, "long")
    expected = 50000 + 50000 * 0.01  # 50500
    assert tp == expected


def test_max_positions_is_5(risk_mgr):
    """Scalping config limits to 5 positions."""
    for i in range(5):
        risk_mgr.register_position(f"COIN{i}/USDT", "long", 100, 1, 95, 110)
    can, reason = risk_mgr.can_open_position("NEW/USDT", 10000)
    assert can is False
