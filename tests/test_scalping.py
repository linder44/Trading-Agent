"""Tests for scalping analysis module."""

import numpy as np
import pandas as pd
import pytest

from analysis.scalping import ScalpingAnalyzer


@pytest.fixture
def analyzer():
    return ScalpingAnalyzer()


@pytest.fixture
def sample_ohlcv():
    """Generate realistic 1m OHLCV data for scalping tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1min")
    price = 50000 + np.cumsum(np.random.randn(n) * 10)
    df = pd.DataFrame({
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 20),
        "low": price - np.abs(np.random.randn(n) * 20),
        "close": price + np.random.randn(n) * 5,
        "volume": np.abs(np.random.randn(n) * 100000 + 500000),
    }, index=dates)
    return df


@pytest.fixture
def small_ohlcv():
    """Generate very small dataset (< 10 rows)."""
    np.random.seed(42)
    n = 5
    price = [50000, 50010, 49990, 50020, 50005]
    df = pd.DataFrame({
        "open": price,
        "high": [p + 15 for p in price],
        "low": [p - 15 for p in price],
        "close": [50005, 50015, 49985, 50025, 50010],
        "volume": [100000] * n,
    })
    return df


def test_full_scalping_analysis(analyzer, sample_ohlcv):
    result = analyzer.full_scalping_analysis(sample_ohlcv)
    assert "order_flow" in result
    assert "micro_momentum" in result
    assert "volume_profile_short" in result
    assert "volatility_regime" in result
    assert "price_action" in result
    assert "spread_estimate" in result
    assert "scalp_signal" in result


def test_full_scalping_insufficient_data(analyzer, small_ohlcv):
    result = analyzer.full_scalping_analysis(small_ohlcv)
    assert "error" in result


def test_order_flow_imbalance(analyzer, sample_ohlcv):
    result = analyzer.order_flow_imbalance(sample_ohlcv)
    assert "imbalance" in result
    assert "signal" in result
    assert -1 <= result["imbalance"] <= 1
    assert result["signal"] in (
        "strong_buy_pressure", "moderate_buy_pressure",
        "strong_sell_pressure", "moderate_sell_pressure", "balanced"
    )


def test_order_flow_small_data(analyzer, small_ohlcv):
    result = analyzer.order_flow_imbalance(small_ohlcv)
    assert "imbalance" in result
    assert "signal" in result


def test_micro_momentum(analyzer, sample_ohlcv):
    result = analyzer.micro_momentum(sample_ohlcv)
    assert "roc_3" in result
    assert "roc_5" in result
    assert "acceleration" in result
    assert "volume_factor" in result
    assert "signal" in result


def test_short_volume_profile(analyzer, sample_ohlcv):
    result = analyzer.short_volume_profile(sample_ohlcv)
    assert "micro_vpoc" in result
    assert "signal" in result
    assert result["micro_vpoc"] > 0


def test_volatility_micro_regime(analyzer, sample_ohlcv):
    result = analyzer.volatility_micro_regime(sample_ohlcv)
    assert "regime" in result
    assert result["regime"] in ("expanding", "contracting", "elevated", "normal", "unknown")
    assert "atr_ratio" in result


def test_price_action_signals(analyzer, sample_ohlcv):
    result = analyzer.price_action_signals(sample_ohlcv)
    assert "patterns" in result
    assert "signal" in result
    assert isinstance(result["patterns"], list)


def test_spread_estimation(analyzer, sample_ohlcv):
    result = analyzer.spread_estimation(sample_ohlcv)
    assert "spread_pct" in result
    assert "signal" in result
    assert result["spread_pct"] >= 0


def test_aggregate_scalp_signal(analyzer, sample_ohlcv):
    result = analyzer._aggregate_scalp_signal(sample_ohlcv)
    assert "score" in result
    assert "verdict" in result
    assert "quality" in result
    assert result["verdict"] in ("strong_buy", "buy", "neutral", "sell", "strong_sell")
    assert result["quality"] in ("good", "risky", "poor")
