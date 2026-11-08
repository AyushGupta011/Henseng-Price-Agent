from datetime import UTC, datetime, timedelta

from domain.enums import LevelType
from domain.models import OHLCVBar
from infrastructure.analysis.support_resistance import SupportResistanceAnalyzer


def test_support_resistance_computation() -> None:
    analyzer = SupportResistanceAnalyzer(order=2, cluster_threshold_pct=0.01)
    base_time = datetime(2025, 5, 1, 9, 30, 0, tzinfo=UTC)

    # Synthetic undulating price pattern: 100 -> 110 -> 95 -> 115 -> 90 -> 105
    prices = [
        100, 103, 107, 110, 106, 102, 98, 95, 99, 104, 109, 115, 112, 105, 97, 90, 96, 101, 105
    ]
    bars = []
    for i, p in enumerate(prices):
        bars.append(
            OHLCVBar(
                ticker="0700.HK",
                timestamp_utc=base_time + timedelta(hours=i),
                open=p - 1.0,
                high=p + 2.0,
                low=p - 2.0,
                close=float(p),
                volume=50000.0,
            )
        )

    levels = analyzer.compute_levels(bars)
    assert len(levels) > 0

    # Verify pivot levels are computed
    types = [lvl.level_type for lvl in levels]
    assert LevelType.PIVOT_PP in types
    assert LevelType.PIVOT_R1 in types
    assert LevelType.PIVOT_S1 in types

    # Verify swing levels
    res_levels = [lvl for lvl in levels if lvl.level_type == LevelType.RESISTANCE]
    sup_levels = [lvl for lvl in levels if lvl.level_type == LevelType.SUPPORT]
    assert len(res_levels) > 0
    assert len(sup_levels) > 0

    # Resistances should generally sit above supports
    assert max(r.price for r in res_levels) > min(s.price for s in sup_levels)
