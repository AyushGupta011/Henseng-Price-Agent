import numpy as np
from scipy.signal import argrelextrema

from domain.enums import LevelType
from domain.models import OHLCVBar, SupportResistanceLevel


class SupportResistanceAnalyzer:
    """Calculates swing extrema (via scipy argrelextrema, clustered) and classical pivot points."""

    def __init__(self, order: int = 3, cluster_threshold_pct: float = 0.008) -> None:
        self.order = order
        self.cluster_threshold_pct = cluster_threshold_pct

    def compute_levels(self, bars: list[OHLCVBar]) -> list[SupportResistanceLevel]:
        if len(bars) < 10:
            return []

        # Sort chronologically
        sorted_bars = sorted(bars, key=lambda b: b.timestamp_utc)
        highs = np.array([b.high for b in sorted_bars], dtype=float)
        lows = np.array([b.low for b in sorted_bars], dtype=float)

        # 1. Classical Pivot Points from last 24 bars (or full session)
        recent_window = sorted_bars[-24:] if len(sorted_bars) >= 24 else sorted_bars
        h_prev = max(b.high for b in recent_window)
        l_prev = min(b.low for b in recent_window)
        c_prev = recent_window[-1].close

        pp = (h_prev + l_prev + c_prev) / 3.0
        r1 = (2.0 * pp) - l_prev
        s1 = (2.0 * pp) - h_prev
        r2 = pp + (h_prev - l_prev)
        s2 = pp - (h_prev - l_prev)

        pivot_levels = [
            SupportResistanceLevel(
                level_type=LevelType.PIVOT_PP, price=round(pp, 2), strength=3.0, method="pivot_point"
            ),
            SupportResistanceLevel(
                level_type=LevelType.PIVOT_R1, price=round(r1, 2), strength=2.0, method="pivot_r1"
            ),
            SupportResistanceLevel(
                level_type=LevelType.PIVOT_S1, price=round(s1, 2), strength=2.0, method="pivot_s1"
            ),
            SupportResistanceLevel(
                level_type=LevelType.PIVOT_R2, price=round(r2, 2), strength=1.5, method="pivot_r2"
            ),
            SupportResistanceLevel(
                level_type=LevelType.PIVOT_S2, price=round(s2, 2), strength=1.5, method="pivot_s2"
            ),
        ]

        # 2. Swing Highs & Lows via scipy argrelextrema
        high_idx = argrelextrema(highs, np.greater, order=self.order)[0]
        low_idx = argrelextrema(lows, np.less, order=self.order)[0]

        raw_resistances = highs[high_idx].tolist()
        raw_supports = lows[low_idx].tolist()

        # 3. Cluster nearby levels within threshold
        clustered_resistances = self._cluster_levels(raw_resistances, LevelType.RESISTANCE)
        clustered_supports = self._cluster_levels(raw_supports, LevelType.SUPPORT)

        # Combine all and sort by price
        all_levels = pivot_levels + clustered_resistances + clustered_supports
        all_levels.sort(key=lambda lvl: lvl.price)
        return all_levels

    def _cluster_levels(
        self, prices: list[float], level_type: LevelType
    ) -> list[SupportResistanceLevel]:
        if not prices:
            return []

        sorted_p = sorted(prices)
        clusters: list[list[float]] = []

        for p in sorted_p:
            if not clusters:
                clusters.append([p])
            else:
                cluster_mean = sum(clusters[-1]) / len(clusters[-1])
                diff_pct = abs(p - cluster_mean) / (cluster_mean + 1e-6)
                if diff_pct <= self.cluster_threshold_pct:
                    clusters[-1].append(p)
                else:
                    clusters.append([p])

        results: list[SupportResistanceLevel] = []
        for c in clusters:
            avg_price = float(np.mean(c))
            strength = float(len(c))  # number of touches
            results.append(
                SupportResistanceLevel(
                    level_type=level_type,
                    price=round(avg_price, 2),
                    strength=round(strength, 1),
                    method="argrelextrema_clustered",
                )
            )

        # Limit to top 3 strongest per type
        results.sort(key=lambda x: x.strength, reverse=True)
        return results[:3]
