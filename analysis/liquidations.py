"""Liquidation data module — fetches recent liquidation events.

Large liquidation cascades indicate forced selling/buying and
often mark local tops/bottoms. Claude uses this to:
- Detect squeeze conditions (mass liquidation = reversal likely)
- Gauge market stress level
- Avoid entering during liquidation cascades
"""

import requests
from loguru import logger


class LiquidationAnalyzer:
    """Fetches liquidation data from Bitget public API."""

    def get_liquidations(self, symbol: str) -> dict:
        """Get recent liquidation data for a symbol.

        Uses Bitget public API for long/short position data as a proxy
        for liquidation pressure (positions being force-closed).
        """
        base_coin = symbol.split("/")[0]
        try:
            # Bitget position long/short — shows aggregate positioning
            # When positions drop sharply, it indicates liquidations
            resp = requests.get(
                "https://api.bitget.com/api/v2/mix/market/account-long-short",
                params={
                    "symbol": f"{base_coin}USDT",
                    "period": "1h",
                    "productType": "USDT-FUTURES",
                },
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data and isinstance(data, list) and len(data) >= 2:
                    latest = data[-1]
                    prev = data[-2]
                    long_now = float(latest.get("longAccountRatio", 0.5))
                    long_prev = float(prev.get("longAccountRatio", 0.5))
                    short_now = float(latest.get("shortAccountRatio", 0.5))
                    short_prev = float(prev.get("shortAccountRatio", 0.5))

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
