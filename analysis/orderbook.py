"""Order book depth analysis — bid/ask imbalance at multiple depth levels.

Complements whale_tracker.py (which detects walls and whale trades) by providing
granular depth-based imbalance metrics that help predict short-term price direction.
"""

import numpy as np
from loguru import logger


class OrderBookAnalyzer:
    """Analyze order book depth for scalping signals."""

    def analyze(self, order_book: dict, current_price: float) -> dict:
        """Full order book analysis: multi-depth imbalance + pressure zones.

        Args:
            order_book: dict with 'bids' and 'asks' lists from ccxt
            current_price: current market price

        Returns:
            dict with imbalance at various depths, pressure signal, absorption zones
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks:
            return {"signal": "no_data", "imbalance_5": 0, "imbalance_10": 0, "imbalance_20": 0}

        # Multi-depth imbalance
        imb_5 = self._depth_imbalance(bids, asks, 5)
        imb_10 = self._depth_imbalance(bids, asks, 10)
        imb_20 = self._depth_imbalance(bids, asks, 20)

        # Weighted imbalance (closer levels matter more)
        weighted_imb = self._weighted_imbalance(bids, asks, current_price)

        # Cumulative depth curve slope
        bid_slope = self._depth_slope(bids, current_price, side="bid")
        ask_slope = self._depth_slope(asks, current_price, side="ask")

        # Pressure zones — where is the most volume concentrated?
        bid_pressure = self._pressure_zone(bids, current_price)
        ask_pressure = self._pressure_zone(asks, current_price)

        # Generate signal
        signal = self._classify_signal(imb_5, imb_10, imb_20, weighted_imb)

        return {
            "imbalance_5": round(imb_5, 3),
            "imbalance_10": round(imb_10, 3),
            "imbalance_20": round(imb_20, 3),
            "weighted_imbalance": round(weighted_imb, 3),
            "bid_depth_slope": round(bid_slope, 4),
            "ask_depth_slope": round(ask_slope, 4),
            "bid_pressure_zone": bid_pressure,
            "ask_pressure_zone": ask_pressure,
            "signal": signal,
        }

    @staticmethod
    def _depth_imbalance(bids: list, asks: list, depth: int) -> float:
        """Bid/ask volume imbalance at given depth level.

        Returns value from -1 (all asks) to +1 (all bids).
        Positive = buy pressure, negative = sell pressure.
        """
        bid_vol = sum(float(b[1]) for b in bids[:depth])
        ask_vol = sum(float(a[1]) for a in asks[:depth])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    @staticmethod
    def _weighted_imbalance(bids: list, asks: list, current_price: float,
                            max_levels: int = 15) -> float:
        """Distance-weighted imbalance — closer levels have more weight.

        Orders near the current price are more likely to be filled and thus
        have more impact on short-term price movement.
        """
        bid_weighted = 0.0
        ask_weighted = 0.0

        for i, (price, size) in enumerate(bids[:max_levels]):
            price, size = float(price), float(size)
            distance = abs(current_price - price) / current_price
            weight = 1.0 / (1.0 + distance * 100)  # Closer = higher weight
            bid_weighted += size * weight

        for i, (price, size) in enumerate(asks[:max_levels]):
            price, size = float(price), float(size)
            distance = abs(price - current_price) / current_price
            weight = 1.0 / (1.0 + distance * 100)
            ask_weighted += size * weight

        total = bid_weighted + ask_weighted
        if total == 0:
            return 0.0
        return (bid_weighted - ask_weighted) / total

    @staticmethod
    def _depth_slope(levels: list, current_price: float, side: str,
                     max_levels: int = 10) -> float:
        """Slope of cumulative depth curve.

        Steep slope = volume concentrated near current price (strong support/resistance).
        Flat slope = volume spread evenly (weak level).
        """
        if len(levels) < 3:
            return 0.0

        cumulative = []
        total = 0.0
        for price, size in levels[:max_levels]:
            total += float(size)
            dist_pct = abs(float(price) - current_price) / current_price * 100
            cumulative.append((dist_pct, total))

        if len(cumulative) < 2:
            return 0.0

        # Simple slope: volume per % distance
        distances = [c[0] for c in cumulative]
        volumes = [c[1] for c in cumulative]

        if distances[-1] == distances[0]:
            return 0.0

        return volumes[-1] / (distances[-1] - distances[0]) if (distances[-1] - distances[0]) > 0 else 0.0

    @staticmethod
    def _pressure_zone(levels: list, current_price: float,
                       max_levels: int = 20) -> dict:
        """Find where the most volume is concentrated.

        Returns the price range with highest volume density.
        """
        if not levels:
            return {"price": 0, "volume": 0, "distance_pct": 0}

        max_vol = 0
        max_price = 0
        for price, size in levels[:max_levels]:
            price, size = float(price), float(size)
            if size > max_vol:
                max_vol = size
                max_price = price

        dist_pct = (max_price - current_price) / current_price * 100 if current_price else 0

        return {
            "price": round(max_price, 6),
            "volume": round(max_vol, 4),
            "distance_pct": round(dist_pct, 3),
        }

    @staticmethod
    def _classify_signal(imb_5: float, imb_10: float, imb_20: float,
                         weighted: float) -> str:
        """Classify order book state into a trading signal."""
        # Strong signals: multiple depths agree
        avg_imb = (imb_5 + imb_10 + imb_20) / 3

        if avg_imb > 0.3 and weighted > 0.2:
            return "strong_bid_dominance"
        elif avg_imb > 0.15 and weighted > 0.1:
            return "moderate_bid_support"
        elif avg_imb < -0.3 and weighted < -0.2:
            return "strong_ask_dominance"
        elif avg_imb < -0.15 and weighted < -0.1:
            return "moderate_ask_pressure"
        elif abs(avg_imb) < 0.05 and abs(weighted) < 0.05:
            return "balanced"
        else:
            return "mixed"
