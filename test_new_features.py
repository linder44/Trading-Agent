"""Tests for all 6 new features.

Tests:
1. Trade history tracker — recording, stats, persistence
2. Liquidation data — parsing, stress levels
3. Cross-symbol correlation — matrix computation, leaders/laggards
4. Time/session context — sessions, weekday, expiry
5. Trailing stop — logic for longs/shorts, direction constraints
6. Volume Profile / VPOC — price bins, value area

Also tests integration: that brain.py receives and includes all new data.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.trade_history import TradeHistoryTracker
from analysis.liquidations import LiquidationAnalyzer
from analysis.cross_correlation import CrossCorrelationAnalyzer
from analysis.time_context import TimeContextAnalyzer
from analysis.technical import TechnicalAnalyzer
from config import TradingConfig
from risk.manager import RiskManager


# ─── Helpers ────────────────────────────────────────────────────────

def make_ohlcv(n=100, base_price=100.0, symbol_seed=0):
    """Generate realistic OHLCV DataFrame."""
    np.random.seed(42 + symbol_seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, base_price * 0.5)  # Prevent negative
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100, 10000, size=n).astype(float)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


# ═══════════════════════════════════════════════════════════════════
# 1. TRADE HISTORY
# ═══════════════════════════════════════════════════════════════════

class TestTradeHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        # Patch HISTORY_FILE to temp
        self.tracker = TradeHistoryTracker()
        self.tracker.HISTORY_FILE = Path(self.tmp.name)
        self.tracker._trades = []

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_record_winning_long(self):
        self.tracker.record_trade("BTC/USDT:USDT", "long", 50000, 51000, 0.1)
        self.assertEqual(len(self.tracker._trades), 1)
        t = self.tracker._trades[0]
        self.assertEqual(t["result"], "win")
        self.assertAlmostEqual(t["pnl_usdt"], 100.0, places=1)
        self.assertAlmostEqual(t["pnl_pct"], 2.0, places=1)

    def test_record_losing_short(self):
        self.tracker.record_trade("ETH/USDT:USDT", "short", 3000, 3100, 1.0)
        t = self.tracker._trades[0]
        self.assertEqual(t["result"], "loss")
        self.assertAlmostEqual(t["pnl_usdt"], -100.0, places=1)

    def test_record_winning_short(self):
        self.tracker.record_trade("SOL/USDT:USDT", "short", 100, 90, 10)
        t = self.tracker._trades[0]
        self.assertEqual(t["result"], "win")
        self.assertAlmostEqual(t["pnl_usdt"], 100.0, places=1)

    def test_breakeven(self):
        self.tracker.record_trade("BTC/USDT:USDT", "long", 50000, 50000, 1)
        self.assertEqual(self.tracker._trades[0]["result"], "breakeven")

    def test_summary_empty(self):
        s = self.tracker.get_summary_for_prompt()
        self.assertEqual(s["total_trades"], 0)

    def test_summary_with_trades(self):
        # 3 wins, 2 losses
        self.tracker.record_trade("BTC/USDT:USDT", "long", 50000, 51000, 0.1)
        self.tracker.record_trade("BTC/USDT:USDT", "long", 50000, 51500, 0.1)
        self.tracker.record_trade("ETH/USDT:USDT", "long", 3000, 3100, 1)
        self.tracker.record_trade("DOGE/USDT:USDT", "long", 0.1, 0.09, 1000)
        self.tracker.record_trade("DOGE/USDT:USDT", "long", 0.1, 0.08, 1000)

        s = self.tracker.get_summary_for_prompt()
        self.assertEqual(s["total_trades"], 5)
        self.assertEqual(s["win_rate_pct"], 60.0)
        self.assertTrue(s["profit_factor"] > 0)
        self.assertTrue(len(s["worst_symbols"]) > 0)
        self.assertTrue(len(s["recent_trades"]) > 0)

    def test_persistence(self):
        self.tracker.record_trade("BTC/USDT:USDT", "long", 50000, 51000, 0.1)
        # Create new tracker pointing to same file
        tracker2 = TradeHistoryTracker()
        tracker2.HISTORY_FILE = Path(self.tmp.name)
        tracker2._load()
        self.assertEqual(len(tracker2._trades), 1)

    def test_max_history_trim(self):
        self.tracker.MAX_HISTORY = 5
        for i in range(10):
            self.tracker.record_trade("BTC/USDT:USDT", "long", 50000, 51000 + i, 0.01)
        self.assertEqual(len(self.tracker._trades), 5)

    def test_summary_has_required_fields(self):
        self.tracker.record_trade("BTC/USDT:USDT", "long", 50000, 51000, 0.1)
        s = self.tracker.get_summary_for_prompt()
        required = ["total_trades", "win_rate_pct", "total_pnl_usdt",
                     "avg_win_usdt", "avg_loss_usdt", "profit_factor",
                     "worst_symbols", "recent_trades"]
        for key in required:
            self.assertIn(key, s, f"Missing key: {key}")


# ═══════════════════════════════════════════════════════════════════
# 2. LIQUIDATION DATA
# ═══════════════════════════════════════════════════════════════════

class TestLiquidations(unittest.TestCase):
    def setUp(self):
        self.analyzer = LiquidationAnalyzer()

    @patch("analysis.liquidations.requests.get")
    def test_normal_market(self, mock_get):
        """No significant position changes = low stress."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"longAccountRatio": "0.500", "shortAccountRatio": "0.500"},
                {"longAccountRatio": "0.501", "shortAccountRatio": "0.499"},
            ]
        }
        mock_get.return_value = mock_resp

        result = self.analyzer.get_liquidations("BTC/USDT:USDT")
        self.assertEqual(result["stress_level"], "low")
        self.assertEqual(result["signal"], "normal")

    @patch("analysis.liquidations.requests.get")
    def test_long_liquidation_cascade(self, mock_get):
        """Sharp drop in long positions = long liquidations."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"longAccountRatio": "0.60", "shortAccountRatio": "0.40"},
                {"longAccountRatio": "0.45", "shortAccountRatio": "0.55"},
            ]
        }
        mock_get.return_value = mock_resp

        result = self.analyzer.get_liquidations("BTC/USDT:USDT")
        self.assertGreater(result["long_liquidation_pressure"], 0)
        self.assertEqual(result["dominant_liquidation"], "longs")

    @patch("analysis.liquidations.requests.get")
    def test_api_failure_returns_default(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        result = self.analyzer.get_liquidations("BTC/USDT:USDT")
        self.assertEqual(result["stress_level"], "unknown")

    @patch("analysis.liquidations.requests.get")
    def test_get_all_liquidations(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"longAccountRatio": "0.50", "shortAccountRatio": "0.50"},
                {"longAccountRatio": "0.50", "shortAccountRatio": "0.50"},
            ]
        }
        mock_get.return_value = mock_resp

        result = self.analyzer.get_all_liquidations(["BTC/USDT:USDT", "ETH/USDT:USDT"])
        self.assertIn("BTC/USDT:USDT", result)
        self.assertIn("ETH/USDT:USDT", result)

    def test_result_has_required_fields(self):
        """Even on failure, result should have all expected keys."""
        with patch("analysis.liquidations.requests.get", side_effect=Exception("fail")):
            result = self.analyzer.get_liquidations("BTC/USDT:USDT")
        required = ["long_liquidation_pressure", "short_liquidation_pressure",
                     "dominant_liquidation", "stress_level", "signal"]
        for key in required:
            self.assertIn(key, result)


# ═══════════════════════════════════════════════════════════════════
# 3. CROSS-SYMBOL CORRELATION
# ═══════════════════════════════════════════════════════════════════

class TestCrossCorrelation(unittest.TestCase):
    def setUp(self):
        self.analyzer = CrossCorrelationAnalyzer()

    def test_two_correlated_symbols(self):
        """Two symbols with similar price movement should show high correlation."""
        np.random.seed(42)
        base = np.cumsum(np.random.randn(100) * 0.5) + 100
        noise = np.random.randn(100) * 0.1

        df1 = pd.DataFrame({"close": base, "open": base, "high": base + 1, "low": base - 1, "volume": 1000})
        df2 = pd.DataFrame({"close": base + noise, "open": base, "high": base + 1, "low": base - 1, "volume": 1000})

        ohlcv_cache = {
            "BTC/USDT:USDT": {"1h": df1},
            "ETH/USDT:USDT": {"1h": df2},
        }
        result = self.analyzer.compute_correlation_matrix(ohlcv_cache, "1h")
        self.assertEqual(result["num_symbols"], 2)
        self.assertTrue(len(result["high_correlations"]) > 0)

    def test_uncorrelated_symbols(self):
        """Two symbols with random movement should show low correlation."""
        df1 = make_ohlcv(100, 100, symbol_seed=0)
        df2 = make_ohlcv(100, 100, symbol_seed=999)  # Different seed

        ohlcv_cache = {
            "BTC/USDT:USDT": {"1h": df1},
            "RAND/USDT:USDT": {"1h": df2},
        }
        result = self.analyzer.compute_correlation_matrix(ohlcv_cache, "1h")
        self.assertEqual(result["num_symbols"], 2)

    def test_single_symbol_insufficient(self):
        df1 = make_ohlcv(100, 100)
        ohlcv_cache = {"BTC/USDT:USDT": {"1h": df1}}
        result = self.analyzer.compute_correlation_matrix(ohlcv_cache, "1h")
        self.assertEqual(result.get("error"), "insufficient_data")

    def test_relative_strength_ordering(self):
        """Symbol with higher return should be leader."""
        # BTC goes up, DOGE goes down
        n = 60
        btc_close = np.linspace(100, 120, n)
        doge_close = np.linspace(100, 80, n)

        df_btc = pd.DataFrame({"close": btc_close, "open": btc_close, "high": btc_close + 1, "low": btc_close - 1, "volume": 1000})
        df_doge = pd.DataFrame({"close": doge_close, "open": doge_close, "high": doge_close + 1, "low": doge_close - 1, "volume": 1000})

        ohlcv_cache = {
            "BTC/USDT:USDT": {"1h": df_btc},
            "DOGE/USDT:USDT": {"1h": df_doge},
        }
        result = self.analyzer.compute_correlation_matrix(ohlcv_cache, "1h")
        self.assertIn("BTC/USDT:USDT", result["leaders"])
        self.assertIn("DOGE/USDT:USDT", result["laggards"])

    def test_result_has_required_fields(self):
        df1 = make_ohlcv(100, 100, 0)
        df2 = make_ohlcv(100, 200, 1)
        ohlcv_cache = {
            "BTC/USDT:USDT": {"1h": df1},
            "ETH/USDT:USDT": {"1h": df2},
        }
        result = self.analyzer.compute_correlation_matrix(ohlcv_cache, "1h")
        for key in ["num_symbols", "high_correlations", "divergent_pairs", "relative_strength", "leaders", "laggards"]:
            self.assertIn(key, result, f"Missing key: {key}")


# ═══════════════════════════════════════════════════════════════════
# 4. TIME / SESSION CONTEXT
# ═══════════════════════════════════════════════════════════════════

class TestTimeContext(unittest.TestCase):
    def setUp(self):
        self.analyzer = TimeContextAnalyzer()

    def test_asian_session(self):
        with patch("analysis.time_context.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 3, 12, 3, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # Directly test the static method
            session = TimeContextAnalyzer._get_trading_session(3)
            self.assertIn("asia", session["active"])
            self.assertNotIn("us", session["active"])

    def test_us_session(self):
        session = TimeContextAnalyzer._get_trading_session(15)
        self.assertIn("us", session["active"])

    def test_europe_us_overlap(self):
        session = TimeContextAnalyzer._get_trading_session(14)
        self.assertIn("europe", session["active"])
        self.assertIn("us", session["active"])
        self.assertEqual(session["overlap"], "europe_us")
        self.assertEqual(session["volatility_expected"], "high")

    def test_off_hours(self):
        session = TimeContextAnalyzer._get_trading_session(23)
        self.assertIn("off_hours", session["active"])

    def test_weekend_context(self):
        # Saturday
        dt = datetime(2024, 3, 16, 12, 0)  # Saturday
        day_ctx = TimeContextAnalyzer._get_day_context(dt)
        self.assertEqual(day_ctx["type"], "weekend")
        self.assertEqual(day_ctx["recommended_size"], "reduced")

    def test_monday_context(self):
        dt = datetime(2024, 3, 11, 12, 0)  # Monday
        day_ctx = TimeContextAnalyzer._get_day_context(dt)
        self.assertEqual(day_ctx["type"], "monday")

    def test_midweek_context(self):
        dt = datetime(2024, 3, 13, 12, 0)  # Wednesday
        day_ctx = TimeContextAnalyzer._get_day_context(dt)
        self.assertEqual(day_ctx["type"], "midweek")

    def test_expiry_context(self):
        # March 2024 last Friday = March 29
        dt = datetime(2024, 3, 27, 12, 0, tzinfo=timezone.utc)
        expiry = self.analyzer._get_expiry_context(dt)
        self.assertTrue(expiry["is_expiry_week"])
        self.assertTrue(expiry["is_quarterly"])  # March is quarterly

    def test_full_context_has_all_fields(self):
        ctx = self.analyzer.get_time_context()
        for key in ["utc_time", "session", "day_of_week", "day_context", "expiry", "hour_utc"]:
            self.assertIn(key, ctx, f"Missing key: {key}")
        self.assertIn("active", ctx["session"])
        self.assertIn("volatility_expected", ctx["session"])

    def test_days_in_month(self):
        self.assertEqual(TimeContextAnalyzer._days_in_month(2024, 2), 29)  # Leap year
        self.assertEqual(TimeContextAnalyzer._days_in_month(2023, 2), 28)
        self.assertEqual(TimeContextAnalyzer._days_in_month(2024, 1), 31)
        self.assertEqual(TimeContextAnalyzer._days_in_month(2024, 12), 31)


# ═══════════════════════════════════════════════════════════════════
# 5. TRAILING STOP
# ═══════════════════════════════════════════════════════════════════

class TestTrailingStop(unittest.TestCase):
    def setUp(self):
        self.cfg = TradingConfig()
        self.risk = RiskManager(self.cfg)

    def test_trailing_stop_long_profit(self):
        """Long position in profit: trailing stop should move up."""
        self.risk.register_position("BTC/USDT:USDT", "long", 50000, 0.1, 48500, 53000)
        # Price moved up to 52000
        new_sl = self.risk.compute_trailing_stop("BTC/USDT:USDT", 52000)
        self.assertIsNotNone(new_sl)
        # New SL = 52000 * (1 - 0.02) = 50960, higher than 48500
        self.assertGreater(new_sl, 48500)
        self.assertAlmostEqual(new_sl, 50960, places=0)

    def test_trailing_stop_long_no_profit(self):
        """Long position not in profit: trailing stop should NOT move."""
        self.risk.register_position("BTC/USDT:USDT", "long", 50000, 0.1, 48500, 53000)
        # Price still below entry
        new_sl = self.risk.compute_trailing_stop("BTC/USDT:USDT", 49000)
        self.assertIsNone(new_sl)

    def test_trailing_stop_long_never_moves_down(self):
        """Even if price drops, SL should not move down."""
        self.risk.register_position("BTC/USDT:USDT", "long", 50000, 0.1, 49000, 53000)
        # Price at entry — new SL would be 49000, same as existing
        new_sl = self.risk.compute_trailing_stop("BTC/USDT:USDT", 50000)
        self.assertIsNone(new_sl)  # 50000*(1-0.02)=49000, not > 49000

    def test_trailing_stop_short_profit(self):
        """Short position in profit: trailing stop should move down."""
        self.risk.register_position("BTC/USDT:USDT", "short", 50000, 0.1, 51500, 47000)
        # Price moved down to 48000
        new_sl = self.risk.compute_trailing_stop("BTC/USDT:USDT", 48000)
        self.assertIsNotNone(new_sl)
        # New SL = 48000 * (1 + 0.02) = 48960, lower than 51500
        self.assertLess(new_sl, 51500)

    def test_trailing_stop_short_no_profit(self):
        """Short not in profit: should NOT move."""
        self.risk.register_position("BTC/USDT:USDT", "short", 50000, 0.1, 51500, 47000)
        new_sl = self.risk.compute_trailing_stop("BTC/USDT:USDT", 51000)
        self.assertIsNone(new_sl)

    def test_trailing_stop_no_position(self):
        new_sl = self.risk.compute_trailing_stop("NONEXIST/USDT:USDT", 50000)
        self.assertIsNone(new_sl)

    def test_check_all_trailing_stops(self):
        self.risk.register_position("BTC/USDT:USDT", "long", 50000, 0.1, 48500, 53000)
        self.risk.register_position("ETH/USDT:USDT", "short", 3000, 1, 3100, 2800)

        prices = {"BTC/USDT:USDT": 52000, "ETH/USDT:USDT": 2800}
        updates = self.risk.check_all_trailing_stops(lambda s: prices[s])

        # BTC should have update (in profit), ETH should have update (in profit)
        symbols_updated = [u["symbol"] for u in updates]
        self.assertIn("BTC/USDT:USDT", symbols_updated)
        self.assertIn("ETH/USDT:USDT", symbols_updated)

    def test_check_trailing_stops_no_updates(self):
        self.risk.register_position("BTC/USDT:USDT", "long", 50000, 0.1, 48500, 53000)
        # Price below entry
        updates = self.risk.check_all_trailing_stops(lambda s: 49000)
        self.assertEqual(len(updates), 0)


# ═══════════════════════════════════════════════════════════════════
# 6. VOLUME PROFILE / VPOC
# ═══════════════════════════════════════════════════════════════════

class TestVolumeProfile(unittest.TestCase):
    def setUp(self):
        self.analyzer = TechnicalAnalyzer()

    def test_vpoc_computation(self):
        df = make_ohlcv(100, 100)
        result = self.analyzer._compute_volume_profile(df)
        self.assertIn("vpoc", result)
        self.assertIn("vah", result)
        self.assertIn("val", result)
        self.assertFalse(np.isnan(result["vpoc"]))
        # VAL <= VPOC <= VAH
        self.assertLessEqual(result["val"], result["vpoc"])
        self.assertLessEqual(result["vpoc"], result["vah"])

    def test_vpoc_in_price_range(self):
        df = make_ohlcv(100, 1000)
        result = self.analyzer._compute_volume_profile(df)
        self.assertGreater(result["vpoc"], df["low"].min())
        self.assertLess(result["vpoc"], df["high"].max())

    def test_vpoc_insufficient_data(self):
        df = make_ohlcv(10, 100)
        result = self.analyzer._compute_volume_profile(df)
        self.assertTrue(np.isnan(result["vpoc"]))

    def test_vpoc_constant_price(self):
        """All candles at same price."""
        df = pd.DataFrame({
            "open": [100.0] * 30,
            "high": [100.0] * 30,
            "low": [100.0] * 30,
            "close": [100.0] * 30,
            "volume": [1000.0] * 30,
        })
        result = self.analyzer._compute_volume_profile(df)
        self.assertEqual(result["vpoc"], 100.0)

    def test_vpoc_in_indicators(self):
        """VPOC columns should appear in compute_indicators output."""
        df = make_ohlcv(100, 50000)
        df_ind = self.analyzer.compute_indicators(df)
        self.assertIn("vpoc", df_ind.columns)
        self.assertIn("vah", df_ind.columns)
        self.assertIn("val", df_ind.columns)
        # Should have actual values (not all NaN)
        self.assertFalse(df_ind["vpoc"].isna().all())

    def test_vpoc_in_summary(self):
        """Volume Profile should appear in generate_summary."""
        df = make_ohlcv(100, 50000)
        df_ind = self.analyzer.compute_indicators(df)
        summary = self.analyzer.generate_summary(df_ind, "BTC/USDT:USDT")
        self.assertIn("vpoc", summary)
        self.assertIn("value_area_high", summary)
        self.assertIn("value_area_low", summary)
        self.assertIn("price_vs_vpoc", summary)


# ═══════════════════════════════════════════════════════════════════
# 7. INTEGRATION: brain.py accepts new data
# ═══════════════════════════════════════════════════════════════════

class TestBrainIntegration(unittest.TestCase):
    """Test that brain._build_prompt includes all new data sections."""

    def setUp(self):
        # Import brain but don't init Claude client
        from agent.brain import TradingBrain
        self.brain = TradingBrain.__new__(TradingBrain)
        # Minimal init
        self.brain.model = "claude-sonnet-4-6"
        self.brain.max_tokens = 16384

    def _build_test_prompt(self, **kwargs):
        defaults = {
            "technical_data": {"BTC/USDT:USDT": {"1h": {"trend": "bullish"}}},
            "market_context": {"fear_greed_index": {"value": 50}},
            "portfolio": {"open_positions": [], "num_positions": 0},
            "balance": 10000,
        }
        defaults.update(kwargs)
        return self.brain._build_prompt(**defaults)

    def test_liquidation_data_in_prompt(self):
        prompt = self._build_test_prompt(
            liquidation_data={"BTC/USDT:USDT": {"stress_level": "high", "signal": "potential_bottom"}}
        )
        self.assertIn("Liquidation Data", prompt)
        self.assertIn("potential_bottom", prompt)

    def test_cross_corr_in_prompt(self):
        prompt = self._build_test_prompt(
            cross_corr_data={"leaders": ["BTC"], "laggards": ["DOGE"], "num_symbols": 5}
        )
        self.assertIn("Cross-Symbol Correlation", prompt)
        self.assertIn("leaders", prompt)

    def test_time_context_in_prompt(self):
        prompt = self._build_test_prompt(
            time_context_data={"session": {"active": ["us"]}, "day_of_week": "Tuesday"}
        )
        self.assertIn("Time & Session Context", prompt)
        self.assertIn("Tuesday", prompt)

    def test_trade_history_in_prompt(self):
        prompt = self._build_test_prompt(
            trade_history_data={
                "total_trades": 10, "win_rate_pct": 60,
                "worst_symbols": [{"symbol": "DOGE/USDT:USDT"}],
                "recent_trades": [],
            }
        )
        self.assertIn("Trade History", prompt)
        self.assertIn("win_rate_pct", prompt)

    def test_trade_history_empty_not_in_prompt(self):
        prompt = self._build_test_prompt(
            trade_history_data={"total_trades": 0, "message": "no history"}
        )
        self.assertNotIn("Trade History", prompt)

    def test_all_new_data_together(self):
        """All 6 features data passed at once — prompt should include all."""
        prompt = self._build_test_prompt(
            liquidation_data={"BTC/USDT:USDT": {"stress_level": "low"}},
            cross_corr_data={"leaders": ["BTC"], "num_symbols": 3},
            time_context_data={"session": {"active": ["europe"]}},
            trade_history_data={"total_trades": 5, "win_rate_pct": 40, "recent_trades": []},
        )
        self.assertIn("Liquidation Data", prompt)
        self.assertIn("Cross-Symbol Correlation", prompt)
        self.assertIn("Time & Session Context", prompt)
        self.assertIn("Trade History", prompt)

    def test_prompt_is_json_serializable(self):
        """Prompt content should not crash on serialization."""
        prompt = self._build_test_prompt(
            liquidation_data={"BTC/USDT:USDT": {"stress_level": "low"}},
            cross_corr_data={"leaders": ["BTC"], "num_symbols": 3,
                             "relative_strength": [{"symbol": "BTC", "return_pct": 5.2}]},
            time_context_data={"session": {"active": ["us"]}, "hour_utc": 15},
            trade_history_data={"total_trades": 1, "win_rate_pct": 100, "recent_trades": []},
        )
        # Should be a valid string, not crash
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
