"""Liquidation data module — estimates liquidation pressure from position changes.

Large liquidation cascades indicate forced selling/buying and
often mark local tops/bottoms. Claude uses this to:
- Detect squeeze conditions (mass liquidation = reversal likely)
- Gauge market stress level
- Avoid entering during liquidation cascades

Uses Bybit v5 public API (account-ratio) — works globally, no keys needed.
Compares the two most recent 5-min snapshots to detect sharp position changes.
"""

from loguru import logger

from utils.http import HttpClientError, request_with_retry

BYBIT_V5 = "https://api.bybit.com/v5/market"


class LiquidationAnalyzer:
    """Estimates liquidation pressure from Bybit long/short position changes."""

    def get_liquidations(self, symbol: str) -> dict:
        """Get recent liquidation pressure for a symbol.

        Fetches 2 most recent 5-minute snapshots of long/short account ratio
        from Bybit. A sharp drop in long% = long liquidations, and vice versa.
        """
        base_coin = symbol.split("/")[0]
        try:
            resp = request_with_retry(
                f"{BYBIT_V5}/account-ratio",
                params={
                    "category": "linear",
                    "symbol": f"{base_coin}USDT",
                    "period": "5min",
                    "limit": "2",
                },
                timeout=10,
            )
            if resp:
                body = resp.json()
                data = body.get("result", {}).get("list", [])
                if data and isinstance(data, list) and len(data) >= 2:
                    # Bybit returns newest first, so [0]=latest, [1]=prev
                    latest = data[0]
                    prev = data[1]
                    long_now = float(latest.get("buyRatio", 0.5))
                    long_prev = float(prev.get("buyRatio", 0.5))
                    short_now = float(latest.get("sellRatio", 0.5))
                    short_prev = float(prev.get("sellRatio", 0.5))

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
