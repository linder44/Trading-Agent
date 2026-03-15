"""Tests for risk management module."""

import pytest
from config import TradingConfig
from risk.manager import RiskManager


@pytest.fixture
def risk_mgr():
    cfg = TradingConfig()
    return RiskManager(cfg)


def test_position_size_calculation(risk_mgr):
    size = risk_mgr.calculate_position_size(
        balance=10000, price=50000, stop_loss_price=48500
    )
    assert size > 0
    # Max risk = 10% of 10000 = 1000. Risk per unit = 1500. Size = 1000/1500 ≈ 0.666
    assert size < 1.0


def test_stop_loss_long(risk_mgr):
    sl = risk_mgr.compute_stop_loss(50000, "long")
    assert sl < 50000


def test_stop_loss_short(risk_mgr):
    sl = risk_mgr.compute_stop_loss(50000, "short")
    assert sl > 50000


def test_take_profit_long(risk_mgr):
    tp = risk_mgr.compute_take_profit(50000, "long")
    assert tp > 50000


def test_take_profit_short(risk_mgr):
    tp = risk_mgr.compute_take_profit(50000, "short")
    assert tp < 50000


def test_can_open_position(risk_mgr):
    can, reason = risk_mgr.can_open_position("BTC/USDT", 10000)
    assert can is True
    assert reason == "OK"


def test_max_positions_limit(risk_mgr):
    for i in range(5):
        risk_mgr.register_position(f"COIN{i}/USDT", "long", 100, 1, 95, 110)

    can, reason = risk_mgr.can_open_position("NEW/USDT", 10000)
    assert can is False
    assert "Max open positions" in reason


def test_duplicate_position_blocked(risk_mgr):
    risk_mgr.register_position("BTC/USDT", "long", 50000, 0.1, 48500, 53000)
    can, reason = risk_mgr.can_open_position("BTC/USDT", 10000)
    assert can is False


def test_close_position_pnl(risk_mgr):
    risk_mgr.register_position("BTC/USDT", "long", 50000, 0.1, 48500, 53000)
    pnl = risk_mgr.close_position("BTC/USDT", 52000)
    # PnL = (52000 - 50000) * 0.1 = 200
    assert pnl == pytest.approx(200.0)
    assert "BTC/USDT" not in risk_mgr.positions
