"""Whale Order Following — track large trades and generate trigger signals.

Strategy: detect abnormally large trades via fetchTrades(), determine whale
entry price and direction, cluster whale orders by price level, and generate
trigger order signals at whale entry levels.
"""

import numpy as np
from loguru import logger


class WhaleTracker:
    """Detect and follow whale/large orders for scalping signals."""

    def __init__(self, size_multiplier: float = 5.0):
        """
        Args:
            size_multiplier: a trade is "whale" if its size > avg * multiplier
        """
        self.size_multiplier = size_multiplier

    def analyze_trades(self, trades: list[dict], current_price: float) -> dict:
        """Analyze recent trades to detect whale activity.

        Args:
            trades: list of trade dicts from exchange (ccxt format)
            current_price: current market price

        Returns:
            dict with whale_trades, clusters, signal, suggested trigger levels
        """
        if not trades or len(trades) < 10:
            return {"whale_trades": [], "clusters": [], "signal": "no_data"}

        # Extract trade sizes (cost = price * amount in USDT)
        sizes = []
        parsed_trades = []
        for t in trades:
            cost = float(t.get("cost", 0))
            if cost == 0:
                price = float(t.get("price", 0))
                amount = float(t.get("amount", 0))
                cost = price * amount
            if cost > 0:
                sizes.append(cost)
                parsed_trades.append({
                    "price": float(t.get("price", 0)),
                    "amount": float(t.get("amount", 0)),
                    "cost": cost,
                    "side": t.get("side", "unknown"),
                    "timestamp": t.get("timestamp", 0),
                })

        if not sizes:
            return {"whale_trades": [], "clusters": [], "signal": "no_data"}

        avg_size = np.mean(sizes)
        threshold = avg_size * self.size_multiplier

        # Filter whale trades
        whale_trades = [t for t in parsed_trades if t["cost"] >= threshold]

        if not whale_trades:
            return {
                "whale_trades": [],
                "clusters": [],
                "signal": "no_whales",
                "avg_trade_size": round(avg_size, 2),
                "whale_threshold": round(threshold, 2),
            }

        # Cluster whale trades by price level
        clusters = self._cluster_by_price(whale_trades, current_price)

        # Generate signal
        signal = self._generate_signal(whale_trades, clusters, current_price)

        return {
            "whale_trades_count": len(whale_trades),
            "total_trades_analyzed": len(parsed_trades),
            "avg_trade_size": round(avg_size, 2),
            "whale_threshold": round(threshold, 2),
            "whale_buy_volume": round(sum(t["cost"] for t in whale_trades if t["side"] == "buy"), 2),
            "whale_sell_volume": round(sum(t["cost"] for t in whale_trades if t["side"] == "sell"), 2),
            "clusters": clusters,
            **signal,
        }

    @staticmethod
    def _cluster_by_price(whale_trades: list[dict], current_price: float,
                          tolerance_pct: float = 0.1) -> list[dict]:
        """Cluster whale trades by price level.

        Groups whale orders that are within tolerance_pct of each other.
        Returns clusters sorted by total volume.
        """
        if not whale_trades:
            return []

        tolerance = current_price * tolerance_pct / 100
        sorted_trades = sorted(whale_trades, key=lambda t: t["price"])

        clusters = []
        current_cluster = [sorted_trades[0]]

        for trade in sorted_trades[1:]:
            if trade["price"] - current_cluster[-1]["price"] <= tolerance:
                current_cluster.append(trade)
            else:
                clusters.append(current_cluster)
                current_cluster = [trade]
        clusters.append(current_cluster)

        result = []
        for cluster in clusters:
            buy_vol = sum(t["cost"] for t in cluster if t["side"] == "buy")
            sell_vol = sum(t["cost"] for t in cluster if t["side"] == "sell")
            avg_price = np.mean([t["price"] for t in cluster])

            dominant_side = "buy" if buy_vol > sell_vol else "sell"
            total_vol = buy_vol + sell_vol

            result.append({
                "price_level": round(float(avg_price), 6),
                "num_trades": len(cluster),
                "total_volume": round(total_vol, 2),
                "buy_volume": round(buy_vol, 2),
                "sell_volume": round(sell_vol, 2),
                "dominant_side": dominant_side,
                "distance_from_current_pct": round(
                    (avg_price - current_price) / current_price * 100, 3
                ),
            })

        # Sort by total volume (biggest clusters first)
        result.sort(key=lambda c: -c["total_volume"])
        return result[:5]  # Top 5 clusters

    @staticmethod
    def _generate_signal(whale_trades: list[dict], clusters: list[dict],
                         current_price: float) -> dict:
        """Generate trading signal based on whale activity."""
        total_buy = sum(t["cost"] for t in whale_trades if t["side"] == "buy")
        total_sell = sum(t["cost"] for t in whale_trades if t["side"] == "sell")
        total = total_buy + total_sell

        if total == 0:
            return {"signal": "neutral", "whale_bias": 0}

        bias = (total_buy - total_sell) / total  # -1 to +1

        # Find strongest cluster near current price
        trigger_levels = []
        for cluster in clusters:
            dist = abs(cluster["distance_from_current_pct"])
            if dist < 0.5:  # Within 0.5% of current price
                trigger_levels.append({
                    "price": cluster["price_level"],
                    "side": "long" if cluster["dominant_side"] == "buy" else "short",
                    "volume": cluster["total_volume"],
                    "distance_pct": cluster["distance_from_current_pct"],
                })

        # Signal determination
        if bias > 0.4:
            signal = "whale_buying"
            hint = "Крупные покупатели доминируют — ищи лонг на уровнях китов"
        elif bias < -0.4:
            signal = "whale_selling"
            hint = "Крупные продавцы доминируют — ищи шорт или избегай лонгов"
        elif abs(bias) < 0.1:
            signal = "whale_neutral"
            hint = "Киты торгуют в обе стороны — нет чёткого направления"
        else:
            signal = "whale_mixed"
            hint = "Слабый перевес китов — подтверди другими индикаторами"

        return {
            "signal": signal,
            "hint": hint,
            "whale_bias": round(bias, 3),
            "whale_buy_pct": round(total_buy / total * 100, 1),
            "whale_sell_pct": round(total_sell / total * 100, 1),
            "trigger_levels": trigger_levels,
        }

    def analyze_order_book_walls(self, order_book: dict, current_price: float,
                                  wall_multiplier: float = 3.0) -> dict:
        """Detect buy/sell walls in the order book.

        A "wall" is an order significantly larger than average at a price level.
        Walls act as support (buy wall) or resistance (sell wall).
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks:
            return {"buy_walls": [], "sell_walls": [], "signal": "no_data"}

        # Analyze bids (buy side)
        bid_sizes = [float(b[1]) for b in bids]
        avg_bid = np.mean(bid_sizes) if bid_sizes else 0
        buy_walls = []
        for price, size in bids:
            price, size = float(price), float(size)
            if size > avg_bid * wall_multiplier:
                buy_walls.append({
                    "price": round(price, 6),
                    "size": round(size, 4),
                    "size_vs_avg": round(size / avg_bid, 1) if avg_bid > 0 else 0,
                    "distance_pct": round((price - current_price) / current_price * 100, 3),
                })

        # Analyze asks (sell side)
        ask_sizes = [float(a[1]) for a in asks]
        avg_ask = np.mean(ask_sizes) if ask_sizes else 0
        sell_walls = []
        for price, size in asks:
            price, size = float(price), float(size)
            if size > avg_ask * wall_multiplier:
                sell_walls.append({
                    "price": round(price, 6),
                    "size": round(size, 4),
                    "size_vs_avg": round(size / avg_ask, 1) if avg_ask > 0 else 0,
                    "distance_pct": round((price - current_price) / current_price * 100, 3),
                })

        # Sort by proximity to current price
        buy_walls.sort(key=lambda w: -w["price"])
        sell_walls.sort(key=lambda w: w["price"])

        # Bid/ask imbalance
        total_bid_vol = sum(bid_sizes[:10])
        total_ask_vol = sum(ask_sizes[:10])
        total = total_bid_vol + total_ask_vol
        imbalance = (total_bid_vol - total_ask_vol) / total if total > 0 else 0

        if imbalance > 0.3:
            signal = "strong_bid_support"
        elif imbalance > 0.1:
            signal = "moderate_bid_support"
        elif imbalance < -0.3:
            signal = "strong_ask_pressure"
        elif imbalance < -0.1:
            signal = "moderate_ask_pressure"
        else:
            signal = "balanced_book"

        return {
            "buy_walls": buy_walls[:3],
            "sell_walls": sell_walls[:3],
            "bid_ask_imbalance": round(imbalance, 3),
            "signal": signal,
        }
