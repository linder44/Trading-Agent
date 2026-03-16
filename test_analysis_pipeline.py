#!/usr/bin/env python3
"""Comprehensive tests for the analysis pipeline.

Verifies that every indicator, pattern, and quant method produces valid data
that Claude will actually receive. Run without exchange connection or API keys.

Usage:
    python test_analysis_pipeline.py                 # Run all tests
    python test_analysis_pipeline.py --group tech    # Technical only
    python test_analysis_pipeline.py --group quant   # Quant only
    python test_analysis_pipeline.py --group pattern  # Patterns only
    python test_analysis_pipeline.py --group risk    # Risk manager only
    python test_analysis_pipeline.py --group prompt  # Prompt assembly only
"""

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


# ═══════════════════════════════════════════════════
# Test data generators
# ═══════════════════════════════════════════════════

def make_ohlcv(n: int = 200, trend: str = "up", volatility: float = 0.02, base_price: float = 50000.0) -> pd.DataFrame:
    """Generate realistic OHLCV data for testing.

    trend: 'up', 'down', 'sideways', 'volatile'
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="h")

    drift = {"up": 0.0003, "down": -0.0003, "sideways": 0.0, "volatile": 0.0}[trend]
    vol = {"up": volatility, "down": volatility, "sideways": volatility * 0.5, "volatile": volatility * 3}[trend]

    returns = np.random.normal(drift, vol, n)
    close = base_price * np.exp(np.cumsum(returns))

    # Generate realistic OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, vol, n)))
    low = close * (1 - np.abs(np.random.normal(0, vol, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.uniform(100, 10000, n) * (1 + np.abs(returns) * 50)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)
    df.index.name = "timestamp"
    return df


def make_small_ohlcv(n: int = 10) -> pd.DataFrame:
    """Generate tiny OHLCV to test edge cases with insufficient data."""
    return make_ohlcv(n=n)


# ═══════════════════════════════════════════════════
# Assertion helpers
# ═══════════════════════════════════════════════════

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def check(self, condition: bool, name: str, detail: str = ""):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            msg = f"FAIL: {name}" + (f" — {detail}" if detail else "")
            self.errors.append(msg)
            logger.error(f"  {msg}")

    def check_not_nan(self, value, name: str):
        if isinstance(value, float):
            self.check(not math.isnan(value), f"{name} is not NaN", f"got NaN")
        else:
            self.check(True, f"{name} is not NaN")

    def check_in_range(self, value, low, high, name: str):
        self.check(
            low <= value <= high,
            f"{name} in [{low}, {high}]",
            f"got {value}",
        )

    def check_has_keys(self, d: dict, keys: list[str], name: str):
        missing = [k for k in keys if k not in d]
        self.check(len(missing) == 0, f"{name} has all keys", f"missing: {missing}")

    def check_no_nans_in_dict(self, d: dict, name: str, path: str = ""):
        """Recursively check that no float values are NaN in a dict."""
        for k, v in d.items():
            full_key = f"{path}.{k}" if path else k
            if isinstance(v, float):
                if math.isnan(v):
                    self.check(False, f"{name}[{full_key}] not NaN", "got NaN")
            elif isinstance(v, dict):
                self.check_no_nans_in_dict(v, name, full_key)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        self.check_no_nans_in_dict(item, name, f"{full_key}[{i}]")
                    elif isinstance(item, float) and math.isnan(item):
                        self.check(False, f"{name}[{full_key}[{i}]] not NaN", "got NaN")


# ═══════════════════════════════════════════════════
# Test groups
# ═══════════════════════════════════════════════════

def test_technical_indicators(r: TestResult):
    """Test TechnicalAnalyzer: compute_indicators + generate_summary."""
    from analysis.technical import TechnicalAnalyzer
    analyzer = TechnicalAnalyzer()

    logger.info("\n[1/4] compute_indicators — полный набор (200 свечей, тренд вверх)")
    df = make_ohlcv(200, "up")
    result = analyzer.compute_indicators(df)

    # Check all expected columns exist
    expected_cols = [
        "ema_9", "ema_21", "ema_50", "ema_200", "sma_50", "sma_200",
        "macd", "macd_signal", "macd_histogram",
        "rsi", "rsi_6", "stoch_rsi_k", "stoch_rsi_d",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
        "atr", "adx", "adx_pos", "adx_neg",
        "obv", "vwap", "volume_sma_20", "volume_ratio",
        "ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conv",
        "pivot", "support_1", "resistance_1",
    ]
    for col in expected_cols:
        r.check(col in result.columns, f"Column '{col}' exists")

    # Check last row has no NaN for key indicators
    last = result.iloc[-1]
    key_indicators = ["ema_9", "ema_21", "ema_50", "rsi", "macd", "atr", "adx", "bb_upper", "bb_lower", "support_1", "resistance_1", "vwap"]
    for ind in key_indicators:
        val = float(last[ind])
        r.check_not_nan(val, f"last row '{ind}'")
        r.check(val != 0, f"last row '{ind}' != 0", f"got {val}")

    # EMA_200 needs 200 candles — check it's valid on last row
    r.check_not_nan(float(last["ema_200"]), "ema_200 with 200 candles")

    # Support < Close < Resistance (generally)
    r.check(
        float(last["support_1"]) < float(last["close"]) < float(last["resistance_1"]),
        "support < price < resistance",
        f"S={last['support_1']:.2f}, P={last['close']:.2f}, R={last['resistance_1']:.2f}",
    )

    # RSI in range 0-100
    r.check_in_range(float(last["rsi"]), 0, 100, "RSI")

    # BB: lower < close < upper (usually)
    r.check(
        float(last["bb_lower"]) < float(last["bb_upper"]),
        "BB lower < BB upper",
    )

    logger.info(f"\n[1/4] compute_indicators — недостаточно данных (30 свечей)")
    df_small = make_ohlcv(30, "up")
    result_small = analyzer.compute_indicators(df_small)
    last_small = result_small.iloc[-1]
    # With < 50 candles, all indicators should be NaN (safe guard)
    r.check(pd.isna(last_small["ema_200"]), "ema_200 is NaN with 30 candles (expected)")
    r.check(pd.isna(last_small["ema_9"]), "ema_9 is NaN with 30 candles (< MIN_CANDLES guard)")
    r.check(pd.isna(last_small["rsi"]), "rsi is NaN with 30 candles (< MIN_CANDLES guard)")

    logger.info(f"\n[1/4] generate_summary — нормальные данные")
    df_full = analyzer.compute_indicators(make_ohlcv(200, "up"))
    summary = analyzer.generate_summary(df_full, "BTC/USDT:USDT")

    expected_keys = [
        "symbol", "price", "change_1h",
        "ema_9", "ema_21", "ema_50", "ema_200",
        "trend_short", "trend_medium", "golden_cross",
        "rsi", "rsi_zone", "macd", "macd_signal", "macd_histogram", "macd_crossover",
        "stoch_rsi_k", "stoch_rsi_d",
        "bb_upper", "bb_lower", "bb_position", "atr", "atr_pct",
        "adx", "trend_strength",
        "volume", "volume_ratio", "obv_rising",
        "support_1", "resistance_1", "vwap", "price_vs_vwap",
        "above_cloud",
    ]
    r.check_has_keys(summary, expected_keys, "generate_summary")
    r.check_no_nans_in_dict(summary, "generate_summary")
    r.check(summary["rsi_zone"] in ("overbought", "oversold", "neutral"), "rsi_zone valid value")
    r.check(summary["trend_short"] in ("bullish", "bearish"), "trend_short valid value")
    r.check(summary["trend_strength"] in ("strong", "weak"), "trend_strength valid value")

    logger.info(f"\n[1/4] generate_summary — недостаточно данных (10 свечей)")
    df_tiny = make_ohlcv(10)
    result_tiny = analyzer.compute_indicators(df_tiny)
    summary_tiny = analyzer.generate_summary(result_tiny, "BTC/USDT:USDT")
    r.check("error" in summary_tiny, "generate_summary returns error for < 50 rows")

    logger.info(f"\n[1/4] multi_timeframe_analysis")
    ohlcv_dict = {
        "1h": analyzer.compute_indicators(make_ohlcv(200, "up")),
        "4h": analyzer.compute_indicators(make_ohlcv(200, "down")),
        "1d": analyzer.compute_indicators(make_ohlcv(200, "sideways")),
    }
    mtf = analyzer.multi_timeframe_analysis(
        {"1h": make_ohlcv(200, "up"), "4h": make_ohlcv(200, "down"), "1d": make_ohlcv(200, "sideways")},
        "BTC/USDT:USDT",
    )
    r.check(set(mtf.keys()) == {"1h", "4h", "1d"}, "MTF has all timeframes")
    for tf in ["1h", "4h", "1d"]:
        r.check("price" in mtf[tf], f"MTF [{tf}] has price")
        r.check_no_nans_in_dict(mtf[tf], f"MTF [{tf}]")


def test_pattern_recognition(r: TestResult):
    """Test PatternRecognizer: candles, fibonacci, divergences."""
    from analysis.technical import TechnicalAnalyzer
    from analysis.patterns import PatternRecognizer

    analyzer = TechnicalAnalyzer()
    patterns = PatternRecognizer()

    logger.info("\n[2/4] detect_patterns — нормальные данные (200 свечей)")
    df = make_ohlcv(200, "volatile")
    detected = patterns.detect_patterns(df)
    r.check(isinstance(detected, list), "detect_patterns returns list")
    if detected:
        r.check_has_keys(detected[0], ["pattern", "signal", "description"], "pattern dict")
        r.check(detected[0]["signal"] in ("bullish", "bearish"), "pattern signal valid")
        logger.info(f"  Найдено паттернов: {len(detected)}")
        for p in detected[:3]:
            logger.info(f"    {p['pattern']}: {p['signal']}")
    else:
        logger.info("  Паттернов не найдено (допустимо для синтетических данных)")

    logger.info("\n[2/4] detect_patterns — мало данных (3 свечи)")
    df_tiny = make_ohlcv(3)
    detected_tiny = patterns.detect_patterns(df_tiny)
    r.check(detected_tiny == [], "detect_patterns returns [] for < 5 rows")

    logger.info("\n[2/4] compute_fibonacci_levels")
    df = make_ohlcv(200, "up")
    fib = patterns.compute_fibonacci_levels(df)
    expected_fib_keys = ["swing_high", "swing_low", "fib_0.236", "fib_0.382", "fib_0.5", "fib_0.618", "fib_0.786", "price_zone"]
    r.check_has_keys(fib, expected_fib_keys, "fibonacci_levels")
    r.check(fib["swing_high"] > fib["swing_low"], "swing_high > swing_low")
    r.check(fib["fib_0.236"] > fib["fib_0.618"], "fib 0.236 > fib 0.618 (retracement order)")
    r.check_no_nans_in_dict(fib, "fibonacci_levels")
    logger.info(f"  Swing: {fib['swing_low']:.0f} — {fib['swing_high']:.0f}, Zone: {fib['price_zone']}")

    logger.info("\n[2/4] compute_fibonacci_levels — flat market (edge case)")
    flat_df = pd.DataFrame({
        "open": [100.0]*50, "high": [100.0]*50, "low": [100.0]*50,
        "close": [100.0]*50, "volume": [1000]*50,
    })
    fib_flat = patterns.compute_fibonacci_levels(flat_df)
    r.check(fib_flat["swing_high"] == fib_flat["swing_low"], "flat market: swing_high == swing_low")

    logger.info("\n[2/4] detect_divergences")
    df = make_ohlcv(200, "volatile")
    df_with_ind = analyzer.compute_indicators(df)
    divs = patterns.detect_divergences(df_with_ind)
    r.check(isinstance(divs, list), "detect_divergences returns list")
    if divs:
        r.check_has_keys(divs[0], ["type", "indicator", "signal", "description"], "divergence dict")
        logger.info(f"  Дивергенций: {len(divs)}")
        for d in divs[:3]:
            logger.info(f"    {d['type']} ({d['indicator']}): {d['signal']}")

    logger.info("\n[2/4] detect_divergences — мало данных (20 свечей)")
    df_small = make_ohlcv(20)
    df_small_ind = analyzer.compute_indicators(df_small)
    divs_small = patterns.detect_divergences(df_small_ind)
    r.check(divs_small == [], "detect_divergences returns [] for < 30 rows")

    logger.info("\n[2/4] get_full_pattern_analysis — полный набор")
    df = analyzer.compute_indicators(make_ohlcv(200, "down"))
    full = patterns.get_full_pattern_analysis(df)
    r.check_has_keys(full, ["candlestick_patterns", "fibonacci_levels", "divergences"], "full_pattern_analysis")
    r.check(isinstance(full["candlestick_patterns"], list), "candlestick_patterns is list")
    r.check(isinstance(full["fibonacci_levels"], dict), "fibonacci_levels is dict")
    r.check(isinstance(full["divergences"], list), "divergences is list")


def test_quant_analysis(r: TestResult):
    """Test QuantAnalyzer: all 10 methods + regime consensus."""
    from analysis.quant import QuantAnalyzer
    quant = QuantAnalyzer()

    df = make_ohlcv(200, "up")
    close = df["close"]

    # 1. Hurst exponent
    logger.info("\n[3/4] hurst_exponent")
    h = quant.hurst_exponent(close)
    r.check_has_keys(h, ["hurst", "regime", "strategy_hint"], "hurst")
    r.check_in_range(h["hurst"], 0, 1, "hurst value")
    r.check(h["regime"] in ("trending", "mean_reverting", "random_walk", "unknown"), "hurst regime valid")
    logger.info(f"  H={h['hurst']}, regime={h['regime']}")

    h_small = quant.hurst_exponent(close.iloc[:20])
    r.check(h_small["regime"] in ("unknown", "trending", "mean_reverting", "random_walk"), "hurst with < 50 data handles gracefully")

    # 2. Z-score
    logger.info("\n[3/4] zscore_analysis")
    z = quant.zscore_analysis(close)
    r.check_has_keys(z, ["zscore", "signal", "action_hint", "mean", "std"], "zscore")
    r.check_not_nan(z["zscore"], "zscore value")
    r.check_not_nan(z["mean"], "zscore mean")
    r.check(z["std"] > 0, "zscore std > 0", f"got {z['std']}")
    logger.info(f"  Z={z['zscore']}, signal={z['signal']}")

    # Edge: constant price
    const = pd.Series([100.0] * 100)
    z_const = quant.zscore_analysis(const)
    r.check(z_const["signal"] == "neutral", "zscore constant price = neutral")

    # 3. Shannon entropy
    logger.info("\n[3/4] shannon_entropy")
    se = quant.shannon_entropy(close)
    r.check_has_keys(se, ["entropy", "normalized_entropy", "signal", "hint"], "shannon_entropy")
    r.check_in_range(se["normalized_entropy"], 0, 1, "normalized_entropy")
    r.check(se["signal"] in ("high_chaos", "moderate_uncertainty", "low_entropy", "unknown"), "entropy signal valid")
    logger.info(f"  H={se['entropy']:.3f}, norm={se['normalized_entropy']:.3f}, signal={se['signal']}")

    se_small = quant.shannon_entropy(close.iloc[:10])
    r.check(se_small["signal"] == "unknown", "shannon entropy < 30 returns = unknown")

    # 4. Kalman filter
    logger.info("\n[3/4] kalman_filter")
    kf = quant.kalman_filter(close)
    r.check_has_keys(kf, ["kalman_price", "price_vs_kalman", "kalman_velocity", "kalman_trend", "signal"], "kalman")
    r.check(kf["kalman_price"] > 0, "kalman_price > 0", f"got {kf['kalman_price']}")
    r.check_not_nan(kf["price_vs_kalman"], "price_vs_kalman")
    r.check(kf["kalman_trend"] in ("bullish", "bearish", "transitioning", "unknown"), "kalman_trend valid")
    logger.info(f"  price={kf['kalman_price']:.2f}, dev={kf['price_vs_kalman']:.2f}%, trend={kf['kalman_trend']}")

    kf_small = quant.kalman_filter(close.iloc[:5])
    r.check(kf_small["kalman_trend"] == "unknown", "kalman < 10 values = unknown")

    # 5. FFT cycles
    logger.info("\n[3/4] fft_cycles")
    fft = quant.fft_cycles(close)
    r.check_has_keys(fft, ["dominant_cycles", "cycle_position"], "fft_cycles")
    r.check(isinstance(fft["dominant_cycles"], list), "fft dominant_cycles is list")
    if fft["dominant_cycles"]:
        r.check_has_keys(fft["dominant_cycles"][0], ["period_candles", "relative_strength_pct"], "fft cycle entry")
        r.check(fft["dominant_cycles"][0]["period_candles"] >= 5, "fft cycle period >= 5")
        logger.info(f"  Dominant cycle: {fft['dominant_cycles'][0]['period_candles']:.0f} candles, position={fft['cycle_position']}")

    fft_small = quant.fft_cycles(close.iloc[:20])
    r.check(fft_small["dominant_cycles"] == [], "fft < 50 values = empty cycles")

    # 6. Linear regression channel
    logger.info("\n[3/4] linear_regression_channel")
    lr = quant.linear_regression_channel(df)
    r.check_has_keys(lr, ["slope_pct_per_candle", "r_squared", "trend_quality", "position_in_channel_sigma", "signal"], "linreg")
    r.check_in_range(lr["r_squared"], 0, 1, "r_squared")
    r.check(lr["trend_quality"] in ("strong_trend", "moderate_trend", "weak/no_trend"), "trend_quality valid")
    r.check_not_nan(lr["position_in_channel_sigma"], "position_in_channel")
    logger.info(f"  R²={lr['r_squared']:.3f}, slope={lr['slope_pct_per_candle']:.4f}%/candle, quality={lr['trend_quality']}")

    lr_small = quant.linear_regression_channel(df.iloc[:10])
    r.check(lr_small["signal"] == "insufficient_data", "linreg < 50 = insufficient_data")

    # 7. Autocorrelation
    logger.info("\n[3/4] autocorrelation_analysis")
    ac = quant.autocorrelation_analysis(close)
    r.check_has_keys(ac, ["lag_1_acf", "significance_threshold", "significant_lags", "signal", "hint"], "autocorrelation")
    r.check_in_range(ac["lag_1_acf"], -1, 1, "lag_1_acf")
    r.check(ac["signal"] in ("momentum_detected", "mean_reversion_detected", "no_serial_correlation"), "acf signal valid")
    logger.info(f"  lag1={ac['lag_1_acf']:.4f}, signal={ac['signal']}, significant_lags={len(ac['significant_lags'])}")

    # 8. Volatility forecast
    logger.info("\n[3/4] volatility_forecast")
    vf = quant.volatility_forecast(close)
    r.check_has_keys(vf, ["ewma_volatility", "historical_volatility", "vol_ratio", "signal", "hint"], "volatility")
    r.check(vf["ewma_volatility"] > 0, "ewma_vol > 0", f"got {vf['ewma_volatility']}")
    r.check(vf["historical_volatility"] > 0, "hist_vol > 0", f"got {vf['historical_volatility']}")
    r.check(vf["signal"] in ("vol_spike", "vol_compression", "vol_above_average", "vol_normal", "insufficient_data"), "vol signal valid")
    logger.info(f"  EWMA={vf['ewma_volatility']:.4f}%, Hist={vf['historical_volatility']:.4f}%, signal={vf['signal']}")

    # 9. Efficiency ratio
    logger.info("\n[3/4] efficiency_ratio")
    er = quant.efficiency_ratio(close)
    r.check_has_keys(er, ["efficiency_ratio", "signal", "hint"], "efficiency_ratio")
    r.check_in_range(er["efficiency_ratio"], 0, 1, "ER value")
    r.check(er["signal"] in ("highly_efficient", "moderately_efficient", "inefficient_choppy", "insufficient_data"), "ER signal valid")
    logger.info(f"  ER={er['efficiency_ratio']:.3f}, signal={er['signal']}")

    # 10. Value at Risk
    logger.info("\n[3/4] value_at_risk")
    var = quant.value_at_risk(close)
    r.check_has_keys(var, ["var_95_pct", "cvar_95_pct", "max_recommended_position_pct", "signal"], "var")
    r.check(var["var_95_pct"] >= 0, "VaR >= 0", f"got {var['var_95_pct']}")
    r.check(var["cvar_95_pct"] >= var["var_95_pct"], "CVaR >= VaR", f"CVaR={var['cvar_95_pct']}, VaR={var['var_95_pct']}")
    r.check(var["signal"] in ("high_risk", "moderate_risk", "low_risk", "insufficient_data"), "VaR signal valid")
    logger.info(f"  VaR95={var['var_95_pct']:.2f}%, CVaR={var['cvar_95_pct']:.2f}%, max_pos={var['max_recommended_position_pct']:.1f}%")

    # 11. Full analysis + regime consensus
    logger.info("\n[3/4] full_quant_analysis + _regime_consensus")
    full = quant.full_quant_analysis(df)
    expected_keys = [
        "hurst_exponent", "zscore", "shannon_entropy", "kalman_filter", "fft_cycles",
        "linear_regression", "autocorrelation", "volatility_forecast",
        "efficiency_ratio", "value_at_risk", "_regime_consensus",
    ]
    r.check_has_keys(full, expected_keys, "full_quant_analysis")

    regime = full["_regime_consensus"]
    r.check_has_keys(regime, ["regime", "confidence", "votes", "recommended_strategy"], "regime_consensus")
    r.check(regime["regime"] in ("trending", "mean_reverting", "random", "high_risk"), "regime valid")
    r.check_in_range(regime["confidence"], 0, 1, "regime confidence")
    logger.info(f"  Regime: {regime['regime']} (confidence={regime['confidence']:.2f})")
    logger.info(f"  Votes: {regime['votes']}")

    # Check no NaN anywhere in the full result
    r.check_no_nans_in_dict(full, "full_quant_analysis")

    # 12. Different market conditions
    for trend_name in ["down", "sideways", "volatile"]:
        logger.info(f"\n[3/4] full_quant_analysis — {trend_name} market")
        df_t = make_ohlcv(200, trend_name)
        full_t = quant.full_quant_analysis(df_t)
        r.check("_regime_consensus" in full_t, f"regime consensus exists for {trend_name}")
        r.check_no_nans_in_dict(full_t, f"full_quant ({trend_name})")
        logger.info(f"  Regime: {full_t['_regime_consensus']['regime']}")


def test_risk_manager(r: TestResult):
    """Test RiskManager: position sizing, SL/TP, portfolio state."""
    from config import TradingConfig
    from risk.manager import RiskManager

    cfg = TradingConfig()
    rm = RiskManager(cfg)

    logger.info("\n[4/4] calculate_position_size")
    size = rm.calculate_position_size(balance=1000.0, price=50000.0, stop_loss_price=49000.0)
    r.check(size > 0, "position size > 0", f"got {size}")
    r.check(size * 50000 <= 1000 * cfg.max_position_pct * 1.01, "position within max_position_pct", f"size*price = {size * 50000:.2f}")

    # Edge: SL = entry price (risk = 0)
    size_zero = rm.calculate_position_size(balance=1000.0, price=50000.0, stop_loss_price=50000.0)
    r.check(size_zero == 0.0, "position size = 0 when SL = entry", f"got {size_zero}")

    logger.info("\n[4/4] compute_stop_loss / compute_take_profit")
    # With ATR
    sl_long = rm.compute_stop_loss(50000.0, "long", atr=500.0)
    tp_long = rm.compute_take_profit(50000.0, "long", atr=500.0)
    r.check(sl_long < 50000, "SL long < entry", f"got {sl_long}")
    r.check(tp_long > 50000, "TP long > entry", f"got {tp_long}")
    r.check(tp_long - 50000 > 50000 - sl_long, "TP distance > SL distance (RR > 1:1)")
    logger.info(f"  Long: entry=50000, SL={sl_long:.0f}, TP={tp_long:.0f}")

    sl_short = rm.compute_stop_loss(50000.0, "short", atr=500.0)
    tp_short = rm.compute_take_profit(50000.0, "short", atr=500.0)
    r.check(sl_short > 50000, "SL short > entry", f"got {sl_short}")
    r.check(tp_short < 50000, "TP short < entry", f"got {tp_short}")
    logger.info(f"  Short: entry=50000, SL={sl_short:.0f}, TP={tp_short:.0f}")

    # Without ATR (uses config %)
    sl_pct = rm.compute_stop_loss(50000.0, "long", atr=None)
    r.check(sl_pct == 50000 * (1 - cfg.stop_loss_pct), "SL % method correct", f"got {sl_pct}")

    logger.info("\n[4/4] can_open_position")
    can, msg = rm.can_open_position("BTC/USDT:USDT", 1000.0)
    r.check(can is True, "can open first position", msg)

    logger.info("\n[4/4] register_position + portfolio_summary")
    pos = rm.register_position("BTC/USDT:USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
    r.check(pos is not None, "position registered")

    summary = rm.get_portfolio_summary()
    r.check_has_keys(summary, ["open_positions", "num_positions", "daily_pnl", "daily_trades"], "portfolio_summary")
    r.check(summary["num_positions"] == 1, "1 position open")
    r.check(len(summary["open_positions"]) == 1, "1 position in list")
    r.check_has_keys(
        summary["open_positions"][0],
        ["symbol", "side", "entry_price", "amount", "stop_loss", "take_profit", "opened_at"],
        "position entry",
    )
    r.check_no_nans_in_dict(summary, "portfolio_summary")

    # Can't open duplicate
    can2, msg2 = rm.can_open_position("BTC/USDT:USDT", 1000.0)
    r.check(can2 is False, "can't open duplicate position", msg2)

    logger.info("\n[4/4] close_position + PnL")
    pnl = rm.close_position("BTC/USDT:USDT", 51000.0)
    r.check(pnl > 0, "PnL > 0 for profitable long", f"got {pnl}")
    r.check(rm.get_portfolio_summary()["num_positions"] == 0, "0 positions after close")

    # Daily loss check
    rm.daily_pnl = -60.0  # simulate -6% on 1000 balance
    can3, msg3 = rm.can_open_position("TEST/USDT:USDT", 1000.0)
    r.check(can3 is False, "blocked by daily loss limit", msg3)
    rm.reset_daily_stats()


def test_prompt_assembly(r: TestResult):
    """Test that the full prompt assembles correctly with all data sections."""
    from analysis.technical import TechnicalAnalyzer
    from analysis.patterns import PatternRecognizer
    from analysis.quant import QuantAnalyzer
    from config import TradingConfig
    from risk.manager import RiskManager
    from agent.brain import _compact_json

    analyzer = TechnicalAnalyzer()
    patterns = PatternRecognizer()
    quant = QuantAnalyzer()
    rm = RiskManager(TradingConfig())

    logger.info("\n[5/5] Собираем полный промпт как в боевом режиме")

    # Simulate 3 symbols × 3 timeframes
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    ohlcv_cache = {}
    technical_data = {}
    pattern_data = {}
    quant_data = {}

    for symbol in symbols:
        ohlcv_dict = {}
        for tf, n in [("1h", 200), ("4h", 200), ("1d", 200)]:
            ohlcv_dict[tf] = make_ohlcv(n, np.random.choice(["up", "down", "sideways"]))

        technical_data[symbol] = analyzer.multi_timeframe_analysis(ohlcv_dict, symbol)
        ohlcv_cache[symbol] = ohlcv_dict

        pattern_data[symbol] = {}
        quant_data[symbol] = {}
        for tf, df in ohlcv_dict.items():
            df_ind = analyzer.compute_indicators(df)
            pattern_data[symbol][tf] = patterns.get_full_pattern_analysis(df_ind)
            quant_data[symbol][tf] = quant.full_quant_analysis(df)

    # Simulate other data
    market_context = {
        "crypto_news": [{"title": "BTC hits 100k", "source": "CoinDesk"}],
        "fear_greed_index": {"value": 23, "classification": "Extreme Fear"},
        "trending_coins": [{"symbol": "BTC", "name": "Bitcoin"}],
    }
    social_data = {
        "social_trending": {"trending_by_social": [{"symbol": "BTC", "score": 1}]},
        "sector_performance": {"sectors": [{"sector": "DeFi", "market_cap_change_24h": 2.5}], "narrative": "DeFi leads"},
    }
    correlation_data = {
        "btc_dominance": {"btc_dominance": 56.6, "signal": "balanced"},
        "stablecoin_market": {"usdt_market_cap": 144.0, "usdc_market_cap": 60.0},
    }
    onchain_data = {
        "BTC/USDT:USDT": {
            "funding_rate": {"funding_rate": 0.0001, "sentiment": "bullish"},
            "open_interest": {"open_interest_value_usd": 15000000000},
            "long_short_ratio": {"ratio": 1.2, "signal": "neutral"},
        },
        "_market_wide": {
            "whale_alerts": [{"symbol": "BTC", "amount_btc": 50, "type": "large_transfer"}],
            "exchange_netflow": {"bid_ratio": 0.55, "signal": "accumulation"},
        },
    }
    portfolio = rm.get_portfolio_summary()

    # Build prompt sections (same logic as brain._build_prompt)
    sections = {}

    # Technical
    tech_json = ""
    for symbol, timeframes in technical_data.items():
        for tf, data in timeframes.items():
            tech_json += _compact_json(data) + "\n"
    sections["technical"] = tech_json

    sections["patterns"] = _compact_json(pattern_data)
    sections["quant"] = _compact_json(quant_data)
    sections["onchain"] = _compact_json(onchain_data)
    sections["news"] = _compact_json(market_context)
    sections["social"] = _compact_json(social_data)
    sections["correlations"] = _compact_json(correlation_data)
    sections["portfolio"] = _compact_json(portfolio)

    total_chars = sum(len(v) for v in sections.values())
    total_tokens_est = total_chars // 3

    logger.info(f"\n  Размеры секций промпта ({len(symbols)} символов × 3 таймфрейма):")
    for name, content in sorted(sections.items(), key=lambda x: -len(x[1])):
        chars = len(content)
        tokens_est = chars // 3
        logger.info(f"    {name:15s}: {chars:>8,} символов (~{tokens_est:>6,} токенов)")

    logger.info(f"    {'ИТОГО':15s}: {total_chars:>8,} символов (~{total_tokens_est:>6,} токенов)")

    # Validate sizes
    r.check(total_tokens_est < 180_000, "total prompt fits in context window (< 180K tokens)", f"got ~{total_tokens_est:,}")
    r.check(len(sections["technical"]) > 0, "technical section non-empty")
    r.check(len(sections["quant"]) > 100, "quant section substantial")
    r.check(len(sections["patterns"]) > 10, "patterns section non-empty")

    # Validate _compact_json removes NaN/None
    logger.info("\n[5/5] _compact_json — NaN/None/empty removal")
    test_data = {
        "normal": 42.123456789,
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "none_val": None,
        "empty_list": [],
        "empty_dict": {},
        "zero_val": 0,
        "nested": {"inner_nan": float("nan"), "good": 1.5},
    }
    compact = _compact_json(test_data)
    parsed = json.loads(compact)
    r.check("nan_val" not in parsed, "NaN removed by _compact_json")
    r.check("inf_val" not in parsed, "Inf removed by _compact_json")
    r.check("none_val" not in parsed, "None removed by _compact_json")
    r.check("empty_list" not in parsed, "empty list removed by _compact_json")
    r.check("empty_dict" not in parsed, "empty dict removed by _compact_json")
    r.check("zero_val" not in parsed, "zero removed by _compact_json")
    r.check("normal" in parsed, "normal value kept")
    r.check(parsed["normal"] == 42.12, "float rounded to 4 significant digits", f"got {parsed['normal']}")
    # Nested NaN should be removed
    r.check("inner_nan" not in parsed.get("nested", {}), "nested NaN removed")

    # Scale test: simulate full 14 symbols × 3 timeframes
    logger.info("\n[5/5] Масштабный тест: 14 символов × 3 таймфрейма")
    full_quant = {}
    full_tech = {}
    full_pattern = {}
    for i in range(14):
        sym = f"SYM{i}/USDT:USDT"
        full_quant[sym] = {}
        full_tech[sym] = {}
        full_pattern[sym] = {}
        for tf in ["1h", "4h", "1d"]:
            df_sim = make_ohlcv(200, np.random.choice(["up", "down", "sideways", "volatile"]))
            full_quant[sym][tf] = quant.full_quant_analysis(df_sim)
            df_ind = analyzer.compute_indicators(df_sim)
            full_tech[sym][tf] = analyzer.generate_summary(df_ind, sym)
            full_pattern[sym][tf] = patterns.get_full_pattern_analysis(df_ind)

    full_quant_json = _compact_json(full_quant)
    full_tech_json = _compact_json(full_tech)
    full_pattern_json = _compact_json(full_pattern)
    total_14 = len(full_quant_json) + len(full_tech_json) + len(full_pattern_json)
    total_14_tokens = total_14 // 3

    logger.info(f"    Technical (14×3): {len(full_tech_json):>8,} символов (~{len(full_tech_json)//3:>6,} токенов)")
    logger.info(f"    Patterns  (14×3): {len(full_pattern_json):>8,} символов (~{len(full_pattern_json)//3:>6,} токенов)")
    logger.info(f"    Quant     (14×3): {len(full_quant_json):>8,} символов (~{len(full_quant_json)//3:>6,} токенов)")
    logger.info(f"    ИТОГО (14×3):     {total_14:>8,} символов (~{total_14_tokens:>6,} токенов)")

    r.check(total_14_tokens < 150_000, "14 symbols × 3 TF fits (< 150K tokens)", f"got ~{total_14_tokens:,}")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test analysis pipeline")
    parser.add_argument("--group", choices=["tech", "pattern", "quant", "risk", "prompt", "all"], default="all")
    args = parser.parse_args()

    groups = {
        "tech": ("TECHNICAL INDICATORS", test_technical_indicators),
        "pattern": ("PATTERN RECOGNITION", test_pattern_recognition),
        "quant": ("QUANTITATIVE ANALYSIS", test_quant_analysis),
        "risk": ("RISK MANAGER", test_risk_manager),
        "prompt": ("PROMPT ASSEMBLY", test_prompt_assembly),
    }

    selected = list(groups.keys()) if args.group == "all" else [args.group]
    total_result = TestResult()

    for key in selected:
        title, test_fn = groups[key]
        logger.info(f"\n{'='*60}")
        logger.info(f"  {title}")
        logger.info(f"{'='*60}")
        try:
            test_fn(total_result)
        except Exception as e:
            total_result.failed += 1
            total_result.errors.append(f"CRASH in {title}: {e}")
            logger.error(f"CRASH: {e}")
            traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("  ИТОГО")
    logger.info(f"{'='*60}")
    logger.info(f"  Пройдено: {total_result.passed}")
    logger.info(f"  Провалено: {total_result.failed}")

    if total_result.errors:
        logger.info(f"\n  Ошибки:")
        for err in total_result.errors:
            logger.error(f"    {err}")

    return 0 if total_result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
