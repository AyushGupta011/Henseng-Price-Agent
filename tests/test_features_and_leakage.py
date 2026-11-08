from datetime import UTC, datetime, timedelta

import numpy as np

from application.features import compute_leakage_free_features
from application.sentiment import ScoredItem
from domain.enums import Exchange
from domain.models import OHLCVBar, Ticker


def test_strict_temporal_alignment_no_future_leakage() -> None:
    """
    CRITICAL TEST: Verifies that a sentiment post at timestamp t
    NEVER leaks into any bar starting at or before t.
    """
    ticker = Ticker(
        symbol="0700.HK",
        exchange=Exchange.HKEX,
        name_en="Tencent",
        name_local="騰訊",
        code="0700",
    )
    base_time = datetime(2025, 5, 1, 10, 0, 0, tzinfo=UTC)

    # 10 hourly bars from 10:00 to 19:00 UTC
    bars = []
    for h in range(15):
        t = base_time + timedelta(hours=h)
        bars.append(
            OHLCVBar(
                ticker="0700.HK",
                timestamp_utc=t,
                open=350.0 + h,
                high=352.0 + h,
                low=349.0 + h,
                close=351.0 + h,
                volume=10000.0,
            )
        )

    # A dramatic bullish shock post published at 14:30 UTC
    # (i.e. AFTER bar 4 which started at 14:00, but BEFORE bar 5 which starts at 15:00)
    shock_time = base_time + timedelta(hours=4, minutes=30)
    shock_item = ScoredItem(
        ticker="0700.HK",
        timestamp_utc=shock_time,
        sentiment=1.0,
        direction="bull",
        confidence=1.0,
        engagement_weight=100.0,
        mentions_position="long",
        language="en",
    )

    df_features = compute_leakage_free_features(bars, [shock_item], ticker, lookback_hours=4)

    assert not df_features.empty

    for _, row in df_features.iterrows():
        bar_start: datetime = row["timestamp_utc"]

        # Bar starting at 14:00 UTC occurred BEFORE the shock at 14:30.
        # Its weighted_sentiment MUST BE 0.0 (NO LEAKAGE!)
        if bar_start <= shock_time:
            assert row["weighted_sentiment"] == 0.0, (
                f"LEAKAGE DETECTED: Bar at {bar_start} used future post at {shock_time}!"
            )
            assert row["mention_count"] == 0, (
                f"LEAKAGE DETECTED: Bar at {bar_start} counted future post at {shock_time}!"
            )

        # Bar starting at 15:00 UTC is AFTER the shock.
        # It SHOULD reflect the shock sentiment.
        if bar_start == base_time + timedelta(hours=5):
            assert row["weighted_sentiment"] > 0.9, (
                f"Expected bar at {bar_start} to capture post at {shock_time}"
            )
            assert row["mention_count"] == 1


def test_target_return_is_strictly_next_bar() -> None:
    """Verifies target_next_return is calculated strictly on the next bar close."""
    ticker = Ticker(
        symbol="0700.HK",
        exchange=Exchange.HKEX,
        name_en="Tencent",
        name_local="騰訊",
        code="0700",
    )
    base_time = datetime(2025, 5, 1, 10, 0, 0, tzinfo=UTC)
    bars = [
        OHLCVBar(
            ticker="0700.HK",
            timestamp_utc=base_time + timedelta(hours=h),
            open=100.0 + h * 10,
            high=105.0 + h * 10,
            low=95.0 + h * 10,
            close=100.0 + h * 10,
            volume=1000.0,
        )
        for h in range(10)
    ]

    df = compute_leakage_free_features(bars, [], ticker)
    # Check that for any row, target_next_return == log(next_close / current_close)
    for _, row in df.iterrows():
        if not np.isnan(row["target_next_return"]):
            expected = np.log(row["next_close"] / row["close"])
            assert np.isclose(row["target_next_return"], expected, atol=1e-6)
