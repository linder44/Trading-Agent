"""Correlation Guard — prevents over-concentrated positions.

Problem: opening long BTC, long ETH, long SOL simultaneously
is NOT 3 positions — it's 3x the same directional bet (corr > 0.8).

Solution:
- Track correlation between open positions
- Block new positions that are highly correlated with existing ones
- Count correlated positions as one for risk limits
"""

import numpy as np
import pandas as pd
from loguru import logger


# Pre-defined correlation groups (updated hourly in real system)
# These are approximate 30-day correlations for major crypto pairs
DEFAULT_CORRELATION_GROUPS = {
    "BTC_GROUP": ["BTC/USDT:USDT", "BTC/USDT"],
    "ETH_GROUP": ["ETH/USDT:USDT", "ETH/USDT"],
    "SOL_GROUP": ["SOL/USDT:USDT", "SOL/USDT"],
    "XRP_GROUP": ["XRP/USDT:USDT", "XRP/USDT"],
    "BNB_GROUP": ["BNB/USDT:USDT", "BNB/USDT"],
    "L1_GROUP": ["ADA/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "SUI/USDT:USDT"],
    "L2_GROUP": ["ARB/USDT:USDT", "OP/USDT:USDT", "MATIC/USDT:USDT"],
    "DEFI_GROUP": ["UNI/USDT:USDT", "AAVE/USDT:USDT", "LINK/USDT:USDT"],
    "MEME_GROUP": ["DOGE/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT", "SHIB/USDT:USDT", "FLOKI/USDT:USDT"],
    "AI_GROUP": ["FET/USDT:USDT", "RENDER/USDT:USDT", "TAO/USDT:USDT"],
}

# High correlation pairs (typically > 0.7)
HIGH_CORR_PAIRS = {
    # BTC коррелирует почти со всем
    frozenset(["BTC/USDT:USDT", "ETH/USDT:USDT"]): 0.85,
    frozenset(["BTC/USDT:USDT", "SOL/USDT:USDT"]): 0.80,
    frozenset(["BTC/USDT:USDT", "BNB/USDT:USDT"]): 0.75,
    frozenset(["BTC/USDT:USDT", "AVAX/USDT:USDT"]): 0.72,
    # ETH и L1/L2
    frozenset(["ETH/USDT:USDT", "SOL/USDT:USDT"]): 0.82,
    frozenset(["ETH/USDT:USDT", "AVAX/USDT:USDT"]): 0.78,
    frozenset(["ETH/USDT:USDT", "ARB/USDT:USDT"]): 0.80,
    frozenset(["ETH/USDT:USDT", "OP/USDT:USDT"]): 0.78,
    frozenset(["ETH/USDT:USDT", "MATIC/USDT:USDT"]): 0.75,
    # L2 между собой
    frozenset(["ARB/USDT:USDT", "OP/USDT:USDT"]): 0.85,
    frozenset(["ARB/USDT:USDT", "MATIC/USDT:USDT"]): 0.72,
    # L1 между собой
    frozenset(["SOL/USDT:USDT", "AVAX/USDT:USDT"]): 0.75,
    frozenset(["SOL/USDT:USDT", "SUI/USDT:USDT"]): 0.78,
    frozenset(["SOL/USDT:USDT", "APT/USDT:USDT"]): 0.74,
    frozenset(["SUI/USDT:USDT", "APT/USDT:USDT"]): 0.80,
    frozenset(["ADA/USDT:USDT", "DOT/USDT:USDT"]): 0.73,
    frozenset(["NEAR/USDT:USDT", "AVAX/USDT:USDT"]): 0.72,
    # Мемкоины
    frozenset(["DOGE/USDT:USDT", "SHIB/USDT:USDT"]): 0.75,
    frozenset(["DOGE/USDT:USDT", "PEPE/USDT:USDT"]): 0.70,
    frozenset(["PEPE/USDT:USDT", "WIF/USDT:USDT"]): 0.72,
    frozenset(["PEPE/USDT:USDT", "FLOKI/USDT:USDT"]): 0.74,
    frozenset(["WIF/USDT:USDT", "FLOKI/USDT:USDT"]): 0.70,
    frozenset(["SHIB/USDT:USDT", "FLOKI/USDT:USDT"]): 0.72,
    # AI токены
    frozenset(["FET/USDT:USDT", "RENDER/USDT:USDT"]): 0.78,
    frozenset(["FET/USDT:USDT", "TAO/USDT:USDT"]): 0.75,
    frozenset(["RENDER/USDT:USDT", "TAO/USDT:USDT"]): 0.73,
    # DeFi
    frozenset(["UNI/USDT:USDT", "AAVE/USDT:USDT"]): 0.74,
    frozenset(["LINK/USDT:USDT", "UNI/USDT:USDT"]): 0.70,
}


class CorrelationGuard:
    """Prevents over-concentrated positions in correlated assets."""

    def __init__(self):
        self._correlation_matrix: dict[frozenset, float] = dict(HIGH_CORR_PAIRS)
        self._last_update: float = 0

    def can_open_position(
        self,
        symbol: str,
        direction: str,
        open_positions: dict,  # {symbol: {"side": "long"/"short"}}
    ) -> tuple[bool, str]:
        """Check if opening a new position would create excessive correlation risk.

        Rules:
        - If same-direction position exists in a correlated asset (corr > 0.7) → block
        - Opposite direction in correlated asset → allow (it's a hedge)
        """
        if not open_positions:
            return True, "OK"

        for existing_symbol, pos_info in open_positions.items():
            existing_side = pos_info.get("side", "")
            corr = self._get_correlation(symbol, existing_symbol)

            if corr > 0.7 and direction == existing_side:
                return False, (
                    f"Blocked: {symbol} {direction} correlates {corr:.2f} with "
                    f"existing {existing_symbol} {existing_side}"
                )

        return True, "OK"

    def get_effective_position_count(self, open_positions: dict) -> int:
        """Count effective positions considering correlation.

        3 correlated long positions count as 3 for risk, not 1.
        This is used for risk limit checks.
        """
        if not open_positions:
            return 0

        symbols = list(open_positions.keys())
        count = len(symbols)

        # Add extra count for highly correlated same-direction positions
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = self._get_correlation(symbols[i], symbols[j])
                side_i = open_positions[symbols[i]].get("side", "")
                side_j = open_positions[symbols[j]].get("side", "")
                if corr > 0.7 and side_i == side_j:
                    count += 1  # Extra penalty for correlated same-direction

        return count

    def update_correlations(self, ohlcv_cache: dict[str, dict[str, pd.DataFrame]]):
        """Update correlation matrix from recent price data.

        Called once per hour (not every cycle).
        """
        close_data = {}
        for symbol, tf_dict in ohlcv_cache.items():
            for tf in ("5m", "1m", "15m"):
                df = tf_dict.get(tf)
                if df is not None and len(df) >= 50:
                    close_data[symbol] = df["close"].values[-50:]
                    break

        if len(close_data) < 2:
            return

        symbols = list(close_data.keys())
        min_len = min(len(v) for v in close_data.values())

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                ret_i = np.diff(close_data[symbols[i]][-min_len:]) / close_data[symbols[i]][-min_len:-1]
                ret_j = np.diff(close_data[symbols[j]][-min_len:]) / close_data[symbols[j]][-min_len:-1]
                corr = np.corrcoef(ret_i, ret_j)[0, 1]
                if not np.isnan(corr):
                    self._correlation_matrix[frozenset([symbols[i], symbols[j]])] = float(corr)

        logger.debug(f"Correlation matrix updated: {len(self._correlation_matrix)} pairs")

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        key = frozenset([symbol1, symbol2])
        if key in self._correlation_matrix:
            return self._correlation_matrix[key]

        # Try with/without :USDT suffix
        s1_alt = symbol1.replace(":USDT", "") if ":USDT" in symbol1 else f"{symbol1}:USDT"
        s2_alt = symbol2.replace(":USDT", "") if ":USDT" in symbol2 else f"{symbol2}:USDT"

        for k in (frozenset([s1_alt, symbol2]), frozenset([symbol1, s2_alt]),
                   frozenset([s1_alt, s2_alt])):
            if k in self._correlation_matrix:
                return self._correlation_matrix[k]

        # Default: assume moderate correlation for crypto
        base1 = symbol1.split("/")[0]
        base2 = symbol2.split("/")[0]
        if base1 == base2:
            return 1.0
        return 0.6  # Default crypto correlation
