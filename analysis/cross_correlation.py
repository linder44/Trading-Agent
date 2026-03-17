"""Cross-symbol correlation matrix.

Shows how symbols move together, helping Claude:
- Avoid over-concentrated positions in correlated assets
- Identify divergences (e.g., ETH lagging BTC = catch-up trade)
- Find hedging opportunities
"""

import numpy as np
import pandas as pd
from loguru import logger


class CrossCorrelationAnalyzer:
    """Computes correlation matrix between trading symbols."""

    def compute_correlation_matrix(
        self, ohlcv_cache: dict[str, dict[str, pd.DataFrame]], timeframe: str = "1h"
    ) -> dict:
        """Compute return correlations between all symbols.

        Args:
            ohlcv_cache: {symbol: {timeframe: DataFrame}} from main loop
            timeframe: which timeframe to use for correlation

        Returns:
            dict with correlation matrix, high correlations, and divergences
        """
        # Extract close prices for each symbol
        close_data = {}
        for symbol, tf_dict in ohlcv_cache.items():
            df = tf_dict.get(timeframe)
            if df is not None and len(df) >= 20:
                close_data[symbol] = df["close"].values[-50:]  # Last 50 candles

        if len(close_data) < 2:
            return {"error": "insufficient_data", "num_symbols": len(close_data)}

        # Align lengths
        min_len = min(len(v) for v in close_data.values())
        symbols = list(close_data.keys())

        # Compute returns
        returns = {}
        for sym in symbols:
            prices = close_data[sym][-min_len:]
            ret = np.diff(prices) / prices[:-1]
            returns[sym] = ret

        # Build correlation matrix
        n = len(symbols)
        corr_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                corr = np.corrcoef(returns[symbols[i]], returns[symbols[j]])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        # Find highly correlated pairs (>0.8) and divergent pairs (<0.2)
        high_corr = []
        divergent = []
        for i in range(n):
            for j in range(i + 1, n):
                c = round(float(corr_matrix[i, j]), 3)
                pair = f"{symbols[i]}-{symbols[j]}"
                if abs(c) > 0.8:
                    high_corr.append({"pair": pair, "correlation": c})
                elif abs(c) < 0.2:
                    divergent.append({"pair": pair, "correlation": c})

        # Relative strength (which symbols outperform over period)
        perf = {}
        for sym in symbols:
            prices = close_data[sym][-min_len:]
            pct_change = (prices[-1] - prices[0]) / prices[0] * 100
            perf[sym] = round(pct_change, 2)

        # Sort by performance
        sorted_perf = sorted(perf.items(), key=lambda x: x[1], reverse=True)

        return {
            "num_symbols": n,
            "high_correlations": high_corr[:10],
            "divergent_pairs": divergent[:10],
            "relative_strength": [
                {"symbol": sym, "return_pct": ret} for sym, ret in sorted_perf
            ],
            "leaders": [sym for sym, _ in sorted_perf[:3]],
            "laggards": [sym for sym, _ in sorted_perf[-3:]],
        }
