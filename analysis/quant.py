"""Quantitative / scientific analysis module.

Indicators and methods grounded in mathematics, statistics,
information theory, and signal processing. Unlike standard TA
indicators (RSI, MACD), these have rigorous theoretical foundations.

References for each method are in docstrings.
"""

import numpy as np
import pandas as pd
from loguru import logger


class QuantAnalyzer:
    """Scientific and mathematical market analysis."""

    # ──────────────────────────────────────────────
    # 1. HURST EXPONENT (Fractal Analysis)
    # ──────────────────────────────────────────────

    @staticmethod
    def hurst_exponent(series: pd.Series, max_lag: int = 100) -> dict:
        """Hurst exponent via Rescaled Range (R/S) analysis.

        Invented by Harold Edwin Hurst (1951) for Nile river flood analysis.
        Applied to finance by Benoit Mandelbrot (fractal theory).

        H > 0.5  → persistent (trending) — momentum strategies work
        H = 0.5  → random walk — no edge, stay out
        H < 0.5  → anti-persistent (mean-reverting) — mean reversion works

        Reference: Mandelbrot & Wallis (1969), "Robustness of the R/S statistic"
        """
        ts = series.dropna().values
        n = len(ts)
        if n < 50:
            return {"hurst": 0.5, "regime": "unknown", "confidence": "low"}

        max_lag = min(max_lag, n // 2)
        lags = range(10, max_lag)
        rs_values = []

        for lag in lags:
            rs_list = []
            for start in range(0, n - lag, lag):
                chunk = ts[start:start + lag]
                mean_chunk = np.mean(chunk)
                deviations = chunk - mean_chunk
                cumulative = np.cumsum(deviations)
                r = np.max(cumulative) - np.min(cumulative)
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_list.append(r / s)
            if rs_list:
                rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

        if len(rs_values) < 5:
            return {"hurst": 0.5, "regime": "unknown", "confidence": "low"}

        log_lags, log_rs = zip(*rs_values)
        # Linear regression: log(R/S) = H * log(n) + c
        coeffs = np.polyfit(log_lags, log_rs, 1)
        h = float(coeffs[0])
        h = np.clip(h, 0.0, 1.0)

        if h > 0.6:
            regime = "trending"
            strategy = "momentum/trend-following works well"
        elif h < 0.4:
            regime = "mean_reverting"
            strategy = "mean-reversion/fade extremes works well"
        else:
            regime = "random_walk"
            strategy = "no statistical edge, reduce exposure"

        return {
            "hurst": round(h, 3),
            "regime": regime,
            "strategy_hint": strategy,
        }

    # ──────────────────────────────────────────────
    # 2. Z-SCORE (Statistical Mean Reversion)
    # ──────────────────────────────────────────────

    @staticmethod
    def zscore_analysis(series: pd.Series, window: int = 50) -> dict:
        """Z-Score: how many standard deviations price is from its mean.

        Based on the Central Limit Theorem and normal distribution.
        Used in statistical arbitrage (Avellaneda & Lee, 2010).

        |Z| > 2.0 → 95% of data is within this range → extreme, likely to revert
        |Z| > 3.0 → 99.7% → very extreme, strong mean-reversion signal

        Reference: Avellaneda & Lee (2010), "Statistical Arbitrage in the U.S. Equities Market"
        """
        ts = series.dropna()
        if len(ts) < window:
            return {"zscore": 0, "signal": "neutral"}

        mean = ts.rolling(window).mean().iloc[-1]
        std = ts.rolling(window).std().iloc[-1]

        if std == 0 or np.isnan(std):
            return {"zscore": 0, "signal": "neutral"}

        z = float((ts.iloc[-1] - mean) / std)

        if z > 3.0:
            signal = "extreme_overbought"
            action = "strong mean-reversion SHORT signal"
        elif z > 2.0:
            signal = "overbought"
            action = "mean-reversion SHORT signal"
        elif z < -3.0:
            signal = "extreme_oversold"
            action = "strong mean-reversion LONG signal"
        elif z < -2.0:
            signal = "oversold"
            action = "mean-reversion LONG signal"
        elif z > 1.0:
            signal = "slightly_elevated"
            action = "caution on longs"
        elif z < -1.0:
            signal = "slightly_depressed"
            action = "watch for long entry"
        else:
            signal = "neutral"
            action = "price near statistical mean"

        return {
            "zscore": round(z, 3),
            "signal": signal,
            "action_hint": action,
            "mean": round(float(mean), 6),
            "std": round(float(std), 6),
        }

    # ──────────────────────────────────────────────
    # 3. SHANNON ENTROPY (Information Theory)
    # ──────────────────────────────────────────────

    @staticmethod
    def shannon_entropy(series: pd.Series, bins: int = 20) -> dict:
        """Shannon entropy of return distribution.

        From Claude Shannon's information theory (1948).
        Measures uncertainty/randomness in the market.

        High entropy → unpredictable, chaotic market → reduce position sizes
        Low entropy  → more predictable/ordered → larger positions possible

        Reference: Shannon (1948), "A Mathematical Theory of Communication"
        """
        returns = series.pct_change().dropna()
        if len(returns) < 30:
            return {"entropy": 0, "signal": "unknown"}

        # Discretize returns into bins
        counts, _ = np.histogram(returns, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Remove zero-probability bins

        # H = -Σ p(x) * log2(p(x))
        entropy = float(-np.sum(probs * np.log2(probs)))

        # Max entropy for uniform distribution = log2(bins)
        max_entropy = np.log2(bins)
        normalized = entropy / max_entropy  # 0-1 scale

        if normalized > 0.85:
            signal = "high_chaos"
            hint = "market is very random/unpredictable — reduce sizes"
        elif normalized > 0.7:
            signal = "moderate_uncertainty"
            hint = "some structure but noisy — standard sizing"
        else:
            signal = "low_entropy"
            hint = "market shows structure/patterns — signals more reliable"

        return {
            "entropy": round(entropy, 3),
            "normalized_entropy": round(normalized, 3),
            "signal": signal,
            "hint": hint,
        }

    # ──────────────────────────────────────────────
    # 4. KALMAN FILTER (Optimal State Estimation)
    # ──────────────────────────────────────────────

    @staticmethod
    def kalman_filter(series: pd.Series) -> dict:
        """Simplified 1D Kalman filter for price smoothing.

        Rudolf Kalman (1960). Used in aerospace, GPS, robotics.
        Mathematically optimal linear filter for noisy data.
        Superior to moving averages: adapts to noise level automatically.

        Returns the filtered price and trend direction.

        Reference: Kalman (1960), "A New Approach to Linear Filtering"
        """
        ts = series.dropna().values
        n = len(ts)
        if n < 10:
            return {"kalman_price": float(ts[-1]), "kalman_trend": "unknown"}

        # State: [price, velocity]
        # Simple 1D implementation
        x = ts[0]  # Initial state estimate
        p = 1.0    # Initial estimate error
        q = 0.01   # Process noise (how much we expect price to change)
        r = 1.0    # Measurement noise (how noisy the data is)

        kalman_prices = np.zeros(n)
        kalman_velocities = np.zeros(n)
        prev_x = x

        for i in range(n):
            # Predict
            x_pred = x
            p_pred = p + q

            # Update
            k = p_pred / (p_pred + r)  # Kalman gain
            x = x_pred + k * (ts[i] - x_pred)
            p = (1 - k) * p_pred

            kalman_velocities[i] = x - prev_x
            prev_x = x
            kalman_prices[i] = x

        current_price = float(ts[-1])
        kalman_price = float(kalman_prices[-1])
        velocity = float(np.mean(kalman_velocities[-5:]))

        if velocity > 0 and current_price > kalman_price:
            trend = "bullish"
        elif velocity < 0 and current_price < kalman_price:
            trend = "bearish"
        else:
            trend = "transitioning"

        deviation_pct = (current_price - kalman_price) / kalman_price * 100

        return {
            "kalman_price": round(kalman_price, 6),
            "price_vs_kalman": round(deviation_pct, 3),
            "kalman_velocity": round(velocity, 6),
            "kalman_trend": trend,
            "signal": "overbought_vs_filter" if deviation_pct > 2 else (
                "oversold_vs_filter" if deviation_pct < -2 else "near_fair_value"
            ),
        }

    # ──────────────────────────────────────────────
    # 5. FFT CYCLE DETECTION (Spectral Analysis)
    # ──────────────────────────────────────────────

    @staticmethod
    def fft_cycles(series: pd.Series, top_n: int = 3) -> dict:
        """Detect dominant cycles using Fast Fourier Transform.

        Jean-Baptiste Fourier (1822) — any signal can be decomposed
        into sum of sine waves. Detects hidden periodicities in price.

        Useful for timing entries: if a 20-period cycle exists,
        expect reversal near cycle highs/lows.

        Reference: Ehlers (2001), "Rocket Science for Traders"
        """
        ts = series.dropna().values
        n = len(ts)
        if n < 50:
            return {"dominant_cycles": [], "signal": "insufficient_data"}

        # Detrend (remove linear trend to isolate cycles)
        x = np.arange(n)
        coeffs = np.polyfit(x, ts, 1)
        detrended = ts - np.polyval(coeffs, x)

        # Apply Hann window to reduce spectral leakage
        window = np.hanning(n)
        windowed = detrended * window

        # FFT
        fft_vals = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n)

        # Skip DC component (index 0) and very low frequencies
        min_period = 5   # Ignore cycles shorter than 5 candles
        max_period = n // 2
        with np.errstate(divide="ignore", invalid="ignore"):
            periods = np.where(freqs > 0, 1.0 / freqs, 0.0)
        valid_mask = (freqs > 0) & (periods >= min_period) & (periods <= max_period)

        if not np.any(valid_mask):
            return {"dominant_cycles": [], "signal": "no_cycles_detected"}

        valid_magnitudes = magnitudes.copy()
        valid_magnitudes[~valid_mask] = 0

        # Find top N peaks
        top_indices = np.argsort(valid_magnitudes)[-top_n:][::-1]
        cycles = []
        for idx in top_indices:
            if freqs[idx] > 0 and valid_magnitudes[idx] > 0:
                period = round(1.0 / freqs[idx], 1)
                strength = round(float(valid_magnitudes[idx] / np.max(magnitudes) * 100), 1)
                cycles.append({
                    "period_candles": period,
                    "relative_strength_pct": strength,
                })

        # Determine where we are in the dominant cycle
        cycle_position = "unknown"
        if cycles:
            dominant_period = int(cycles[0]["period_candles"])
            if dominant_period > 0:
                position_in_cycle = n % dominant_period
                pct_through = position_in_cycle / dominant_period
                if pct_through < 0.25:
                    cycle_position = "cycle_bottom_zone"
                elif pct_through < 0.5:
                    cycle_position = "cycle_rising"
                elif pct_through < 0.75:
                    cycle_position = "cycle_top_zone"
                else:
                    cycle_position = "cycle_falling"

        return {
            "dominant_cycles": cycles,
            "cycle_position": cycle_position,
        }

    # ──────────────────────────────────────────────
    # 6. LINEAR REGRESSION CHANNEL
    # ──────────────────────────────────────────────

    @staticmethod
    def linear_regression_channel(df: pd.DataFrame, window: int = 50) -> dict:
        """Linear regression channel with standard deviation bands.

        Based on Ordinary Least Squares (Gauss, Legendre ~1800).
        The regression line = mathematically optimal trend line.
        Deviation from regression = statistically significant move.

        Reference: Gauss (1809), "Theoria Motus"; applied to finance
        by Raschke & Connors (1995), "Street Smarts"
        """
        close = df["close"].dropna().values
        if len(close) < window:
            return {"slope": 0, "r_squared": 0, "signal": "insufficient_data"}

        y = close[-window:]
        x = np.arange(window)

        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Predicted values
        y_pred = np.polyval(coeffs, x)

        # R-squared (coefficient of determination)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Standard deviation of residuals
        residuals = y - y_pred
        std_residual = np.std(residuals)

        # Current position relative to channel
        current_residual = float(y[-1] - y_pred[-1])
        position_in_channel = current_residual / std_residual if std_residual > 0 else 0

        # Slope as percentage per candle
        slope_pct = (slope / y_pred[-1]) * 100 if y_pred[-1] != 0 else 0

        # Channel boundaries at current point
        upper = float(y_pred[-1] + 2 * std_residual)
        lower = float(y_pred[-1] - 2 * std_residual)

        if position_in_channel > 2:
            signal = "above_upper_band"
            hint = "price above +2σ regression channel — overextended"
        elif position_in_channel < -2:
            signal = "below_lower_band"
            hint = "price below -2σ regression channel — undervalued statistically"
        elif position_in_channel > 1:
            signal = "upper_half"
            hint = "in upper part of channel"
        elif position_in_channel < -1:
            signal = "lower_half"
            hint = "in lower part of channel"
        else:
            signal = "near_regression"
            hint = "near regression line (fair value)"

        return {
            "slope_pct_per_candle": round(slope_pct, 4),
            "r_squared": round(r_squared, 3),
            "trend_quality": "strong_trend" if r_squared > 0.8 else (
                "moderate_trend" if r_squared > 0.5 else "weak/no_trend"
            ),
            "position_in_channel_sigma": round(position_in_channel, 2),
            "channel_upper": round(upper, 6),
            "channel_lower": round(lower, 6),
            "regression_price": round(float(y_pred[-1]), 6),
            "signal": signal,
            "hint": hint,
        }

    # ──────────────────────────────────────────────
    # 7. AUTOCORRELATION (Serial Correlation)
    # ──────────────────────────────────────────────

    @staticmethod
    def autocorrelation_analysis(series: pd.Series, max_lag: int = 20) -> dict:
        """Autocorrelation of returns — do past returns predict future?

        Positive autocorrelation → momentum (trend continues)
        Negative autocorrelation → mean reversion (trend reverses)
        Zero → random walk (no edge)

        Ljung-Box test significance threshold: |acf| > 2/√N

        Reference: Box & Jenkins (1976), "Time Series Analysis"
        """
        returns = series.pct_change().dropna()
        n = len(returns)
        if n < 50:
            return {"autocorrelations": {}, "signal": "insufficient_data"}

        mean_ret = returns.mean()
        var = ((returns - mean_ret) ** 2).sum()

        threshold = 2.0 / np.sqrt(n)  # Statistical significance

        acf_results = {}
        significant_lags = []

        for lag in range(1, min(max_lag + 1, n // 4)):
            cov = ((returns.iloc[lag:].values - mean_ret) *
                   (returns.iloc[:-lag].values - mean_ret)).sum()
            acf = float(cov / var) if var > 0 else 0
            acf_results[f"lag_{lag}"] = round(acf, 4)

            if abs(acf) > threshold:
                direction = "momentum" if acf > 0 else "mean_reverting"
                significant_lags.append({
                    "lag": lag,
                    "acf": round(acf, 4),
                    "type": direction,
                })

        # Summarize
        lag1 = acf_results.get("lag_1", 0)
        if lag1 > threshold:
            signal = "momentum_detected"
            hint = f"lag-1 autocorrelation {lag1} > threshold {threshold:.3f} — momentum present"
        elif lag1 < -threshold:
            signal = "mean_reversion_detected"
            hint = f"lag-1 autocorrelation {lag1} < -{threshold:.3f} — mean reversion present"
        else:
            signal = "no_serial_correlation"
            hint = "returns appear random at lag-1 — no momentum or MR edge"

        return {
            "lag_1_acf": round(lag1, 4),
            "significance_threshold": round(threshold, 4),
            "significant_lags": significant_lags[:5],
            "signal": signal,
            "hint": hint,
        }

    # ──────────────────────────────────────────────
    # 8. VOLATILITY CLUSTERING (EWMA Variance)
    # ──────────────────────────────────────────────

    @staticmethod
    def volatility_forecast(series: pd.Series, span: int = 30) -> dict:
        """Exponentially Weighted Moving Average (EWMA) volatility forecast.

        Based on RiskMetrics (J.P. Morgan, 1994) and related to GARCH
        (Bollerslev, 1986). Captures volatility clustering — the empirical
        fact that large moves tend to be followed by large moves.

        Used for dynamic position sizing and risk management.

        Reference: Bollerslev (1986), "Generalized Autoregressive Conditional
        Heteroskedasticity"; J.P. Morgan (1994), RiskMetrics Technical Document
        """
        returns = series.pct_change().dropna()
        if len(returns) < 30:
            return {"forecast_vol": 0, "signal": "insufficient_data"}

        # EWMA variance (λ = 0.94 is the RiskMetrics standard)
        lam = 0.94
        variance = float(returns.iloc[0] ** 2)
        for ret in returns.values[1:]:
            variance = lam * variance + (1 - lam) * (ret ** 2)

        ewma_vol = float(np.sqrt(variance))

        # Compare to historical realized vol
        hist_vol = float(returns.rolling(span).std().iloc[-1])

        # Annualized (assuming ~365 candles/year for crypto)
        annual_factor = np.sqrt(365 * 24)  # Adjust based on timeframe
        annual_vol = ewma_vol * annual_factor

        if ewma_vol > hist_vol * 1.5:
            signal = "vol_spike"
            hint = "volatility expanding rapidly — reduce position sizes, tighten stops"
        elif ewma_vol < hist_vol * 0.5:
            signal = "vol_compression"
            hint = "volatility very low — breakout likely, prepare entries"
        elif ewma_vol > hist_vol:
            signal = "vol_above_average"
            hint = "elevated volatility — use tighter stops"
        else:
            signal = "vol_normal"
            hint = "volatility within normal range"

        return {
            "ewma_volatility": round(ewma_vol * 100, 4),
            "historical_volatility": round(hist_vol * 100, 4),
            "vol_ratio": round(ewma_vol / hist_vol if hist_vol > 0 else 1, 2),
            "signal": signal,
            "hint": hint,
        }

    # ──────────────────────────────────────────────
    # 9. MARKET EFFICIENCY RATIO
    # ──────────────────────────────────────────────

    @staticmethod
    def efficiency_ratio(series: pd.Series, window: int = 20) -> dict:
        """Kaufman's Efficiency Ratio (ER).

        ER = |net price change| / sum of |individual changes|

        ER → 1: price moved efficiently (trending) — trend-follow
        ER → 0: price moved noisily (choppy) — mean-revert or stay out

        Used in Kaufman's Adaptive Moving Average (KAMA).

        Reference: Kaufman (1995), "Smarter Trading"
        """
        ts = series.dropna().values
        if len(ts) < window + 1:
            return {"efficiency_ratio": 0, "signal": "insufficient_data"}

        recent = ts[-window - 1:]

        # Net change (signal)
        net_change = abs(recent[-1] - recent[0])

        # Sum of absolute individual changes (noise)
        noise = sum(abs(recent[i] - recent[i - 1]) for i in range(1, len(recent)))

        er = float(net_change / noise) if noise > 0 else 0

        if er > 0.6:
            signal = "highly_efficient"
            hint = "price trending cleanly — trend-following optimal"
        elif er > 0.3:
            signal = "moderately_efficient"
            hint = "some trend but noisy — standard approach"
        else:
            signal = "inefficient_choppy"
            hint = "very choppy/ranging — avoid trend entries, consider mean reversion"

        return {
            "efficiency_ratio": round(er, 3),
            "signal": signal,
            "hint": hint,
        }

    # ──────────────────────────────────────────────
    # 10. VALUE AT RISK (VaR)
    # ──────────────────────────────────────────────

    @staticmethod
    def value_at_risk(series: pd.Series, confidence: float = 0.95, horizon: int = 1) -> dict:
        """Historical Value at Risk (VaR) and Conditional VaR (CVaR/ES).

        VaR = maximum expected loss at given confidence level.
        CVaR (Expected Shortfall) = average loss when VaR is breached.

        CVaR is considered superior by Basel III banking regulations
        because it captures tail risk.

        Reference: Artzner et al. (1999), "Coherent Measures of Risk"
        """
        returns = series.pct_change().dropna()
        if len(returns) < 30:
            return {"var_95": 0, "cvar_95": 0, "signal": "insufficient_data"}

        sorted_returns = np.sort(returns.values)
        cutoff_idx = int(len(sorted_returns) * (1 - confidence))

        var = float(-sorted_returns[cutoff_idx]) * np.sqrt(horizon)
        cvar = float(-np.mean(sorted_returns[:cutoff_idx + 1])) * np.sqrt(horizon)

        current_price = float(series.iloc[-1])
        var_usd_per_1000 = round(var * 1000, 2)
        cvar_usd_per_1000 = round(cvar * 1000, 2)

        if var > 0.05:
            signal = "high_risk"
            hint = f"95% VaR is {var*100:.1f}% — high risk, reduce exposure"
        elif var > 0.03:
            signal = "moderate_risk"
            hint = f"95% VaR is {var*100:.1f}% — moderate risk"
        else:
            signal = "low_risk"
            hint = f"95% VaR is {var*100:.1f}% — relatively calm"

        return {
            "var_95_pct": round(var * 100, 2),
            "cvar_95_pct": round(cvar * 100, 2),
            "var_usd_per_1000": var_usd_per_1000,
            "cvar_usd_per_1000": cvar_usd_per_1000,
            "max_recommended_position_pct": round(min(5.0 / (var * 100), 10.0), 1) if var > 0 else 10.0,
            "signal": signal,
            "hint": hint,
        }

    # ──────────────────────────────────────────────
    # COMBINED ANALYSIS
    # ──────────────────────────────────────────────

    def full_quant_analysis(self, df: pd.DataFrame, short_term: bool = True) -> dict:
        """Run all quantitative analyses on a DataFrame.

        Args:
            short_term: если True, использует укороченные окна для скальпинга.
        """
        close = df["close"]

        # Короткие окна для скальпинга
        z_window = 20 if short_term else 50
        lr_window = 20 if short_term else 50
        er_window = 10 if short_term else 20
        vol_span = 15 if short_term else 30

        result = {
            "hurst_exponent": self.hurst_exponent(close, max_lag=50 if short_term else 100),
            "zscore": self.zscore_analysis(close, window=z_window),
            "shannon_entropy": self.shannon_entropy(close, bins=15 if short_term else 20),
            "kalman_filter": self.kalman_filter(close),
            "fft_cycles": self.fft_cycles(close),
            "linear_regression": self.linear_regression_channel(df, window=lr_window),
            "autocorrelation": self.autocorrelation_analysis(close, max_lag=10 if short_term else 20),
            "volatility_forecast": self.volatility_forecast(close, span=vol_span),
            "efficiency_ratio": self.efficiency_ratio(close, window=er_window),
            "value_at_risk": self.value_at_risk(close),
        }

        # Meta-analysis: combine signals for overall regime assessment
        result["_regime_consensus"] = self._regime_consensus(result)

        return result

    def _regime_consensus(self, analysis: dict) -> dict:
        """Combine all scientific indicators into a regime consensus."""
        votes = {"trending": 0, "mean_reverting": 0, "random": 0, "high_risk": 0}

        # Hurst
        h = analysis["hurst_exponent"].get("regime", "")
        if h == "trending":
            votes["trending"] += 2
        elif h == "mean_reverting":
            votes["mean_reverting"] += 2
        else:
            votes["random"] += 1

        # Efficiency ratio
        er_sig = analysis["efficiency_ratio"].get("signal", "")
        if "highly" in er_sig:
            votes["trending"] += 1
        elif "choppy" in er_sig:
            votes["mean_reverting"] += 1

        # Autocorrelation
        ac_sig = analysis["autocorrelation"].get("signal", "")
        if "momentum" in ac_sig:
            votes["trending"] += 1
        elif "mean_reversion" in ac_sig:
            votes["mean_reverting"] += 1

        # Linear regression R²
        r2 = analysis["linear_regression"].get("r_squared", 0)
        if r2 > 0.7:
            votes["trending"] += 1
        elif r2 < 0.3:
            votes["mean_reverting"] += 1

        # Volatility
        vol_sig = analysis["volatility_forecast"].get("signal", "")
        if "spike" in vol_sig:
            votes["high_risk"] += 2

        # Entropy
        ent_sig = analysis["shannon_entropy"].get("signal", "")
        if "chaos" in ent_sig:
            votes["high_risk"] += 1
            votes["random"] += 1

        # VaR
        var_sig = analysis["value_at_risk"].get("signal", "")
        if "high" in var_sig:
            votes["high_risk"] += 1

        # Determine winner
        regime = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[regime] / total_votes if total_votes > 0 else 0

        strategy_map = {
            "trending": "Use momentum/trend-following strategies (EMA crossovers, breakouts)",
            "mean_reverting": "Use mean-reversion strategies (buy oversold, sell overbought, fade extremes)",
            "random": "Market is random — reduce exposure, no edge available",
            "high_risk": "High risk regime — reduce all positions, tighten stops, consider hedging",
        }

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "votes": votes,
            "recommended_strategy": strategy_map[regime],
        }
