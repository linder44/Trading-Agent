"""Tests for technical analysis module."""

import numpy as np
import pandas as pd
import pytest

from analysis.technical import TechnicalAnalyzer


@pytest.fixture
def analyzer():
    return TechnicalAnalyzer()


@pytest.fixture
def sample_ohlcv():
    """Generate realistic OHLCV data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    price = 50000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 200),
        "low": price - np.abs(np.random.randn(n) * 200),
        "close": price + np.random.randn(n) * 50,
        "volume": np.abs(np.random.randn(n) * 1000000 + 5000000),
    }, index=dates)
    df.index.name = "timestamp"
    return df


def test_compute_indicators(analyzer, sample_ohlcv):
    result = analyzer.compute_indicators(sample_ohlcv)
    assert "ema_9" in result.columns
    assert "rsi" in result.columns
    assert "macd" in result.columns
    assert "bb_upper" in result.columns
    assert "atr" in result.columns
    assert "adx" in result.columns
    assert "obv" in result.columns
    assert "ichimoku_a" in result.columns


def test_generate_summary(analyzer, sample_ohlcv):
    df = analyzer.compute_indicators(sample_ohlcv)
    summary = analyzer.generate_summary(df, "BTC/USDT")
    assert summary["symbol"] == "BTC/USDT"
    assert "price" in summary
    assert "rsi" in summary
    assert "trend_short" in summary
    assert summary["trend_short"] in ("bullish", "bearish")
    assert 0 <= summary["rsi"] <= 100


def test_multi_timeframe(analyzer, sample_ohlcv):
    ohlcv_dict = {"1h": sample_ohlcv, "4h": sample_ohlcv}
    result = analyzer.multi_timeframe_analysis(ohlcv_dict, "BTC/USDT")
    assert "1h" in result
    assert "4h" in result
