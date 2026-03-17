"""Liquidation data module — estimates liquidation pressure from position changes.

Large liquidation cascades indicate forced selling/buying and
often mark local tops/bottoms. Claude uses this to:
- Detect squeeze conditions (mass liquidation = reversal likely)
- Gauge market stress level
- Avoid entering during liquidation cascades

Uses Binance Futures globalLongShortAccountRatio (public, no keys needed).
Compares the two most recent 5m snapshots to detect sharp position changes.
"""

from loguru import logger

from utils.http import HttpClientError, request_with_retry

BINANCE_FUTURES = "https://fapi.binance.com/futures/data"


class LiquidationAnalyzer:
    """Estimates liquidation pressure from Binance long/short position changes."""

    def get_liquidations(self, symbol: str) -> dict:
        """Get recent liquidation pressure for a symbol.

        Fetches 2 most recent 5-minute snapshots of global long/short ratio
        from Binance. A sharp drop in long% = long liquidations, and vice versa.
        """
        base_coin = symbol.split("/")[0]
        try:
            resp = request_with_retry(
                f"{BINANCE_FUTURES}/globalLongShortAccountRatio",
                params={
                    "symbol": f"{base_coin}USDT",
                    "period": "5m",
                    "limit": "2",
                },
                timeout=10,
            )
            if resp:
                data = resp.json()
                if data and isinstance(data, list) and len(data) >= 2:
                    prev = data[0]
                    latest = data[1]
                    long_now = float(latest.get("longAccount", 0.5))
                    long_prev = float(prev.get("longAccount", 0.5))
                    short_now = float(latest.get("shortAccount", 0.5))
                    short_prev = float(prev.get("shortAccount", 0.5))

                    # Sharp drop in long positions = long liquidations
                    long_liq_pressure = max(0, (long_prev - long_now) * 100)
                    # Sharp drop in short positions = short liquidations
                    short_liq_pressure = max(0, (short_prev - short_now) * 100)

                    total_pressure = long_liq_pressure + short_liq_pressure

                    return {
                        "long_liquidation_pressure": round(long_liq_pressure, 2),
                        "short_liquidation_pressure": round(short_liq_pressure, 2),
                        "dominant_liquidation": "longs" if long_liq_pressure > short_liq_pressure else "shorts",
                        "stress_level": (
                            "extreme" if total_pressure > 5 else
                            "high" if total_pressure > 2 else
                            "moderate" if total_pressure > 0.5 else
                            "low"
                        ),
                        "signal": (
                            "potential_bottom" if long_liq_pressure > 3 else
                            "potential_top" if short_liq_pressure > 3 else
                            "normal"
                        ),
                    }
        except HttpClientError as e:
            logger.warning(f"Liquidation data 4xx for {base_coin}: {e}")
        except Exception as e:
            logger.warning(f"Liquidation data fetch failed for {base_coin}: {e}")

        return {
            "long_liquidation_pressure": 0,
            "short_liquidation_pressure": 0,
            "dominant_liquidation": "none",
            "stress_level": "unknown",
            "signal": "no_data",
        }

    def get_all_liquidations(self, symbols: list[str]) -> dict:
        """Get liquidation data for all symbols."""
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_liquidations(symbol)
        return result
