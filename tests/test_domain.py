from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from domain.constants import HK_TZ, TICKERS_BY_EXCHANGE
from domain.enums import Direction, Exchange, LevelType, PositionClaim
from domain.models import (
    CrowdStance,
    OHLCVBar,
    Post,
    SentimentScore,
    SupportResistanceLevel,
    Ticker,
    TickerSnapshot,
    compute_content_hash,
)


def test_ticker_model() -> None:
    ticker = Ticker(
        symbol="0700.HK",
        exchange=Exchange.HKEX,
        name_en="Tencent Holdings",
        name_local="騰訊控股",
        code="0700",
        search_keywords=["0700", "Tencent"],
    )
    assert ticker.symbol == "0700.HK"
    assert ticker.exchange == Exchange.HKEX
    assert "Tencent" in ticker.search_keywords


def test_post_utc_tz_enforcement() -> None:
    # Naive datetime should fail validation
    naive_dt = datetime(2025, 5, 1, 12, 0, 0)
    with pytest.raises(ValidationError):
        Post(
            source="lihkg",
            ticker="0700.HK",
            text="Tencent looking bullish",
            created_at_utc=naive_dt,
        )

    # Hong Kong local time should be converted to UTC
    hk_dt = datetime(2025, 5, 1, 20, 0, 0, tzinfo=HK_TZ)
    post = Post(
        source="lihkg",
        ticker="0700.HK",
        text="Tencent looking bullish",
        created_at_utc=hk_dt,
    )
    assert post.created_at_utc.tzinfo == UTC
    assert post.created_at_utc.hour == 12  # 20:00 HKT (UTC+8) -> 12:00 UTC


def test_post_engagement_weight() -> None:
    dt = datetime(2025, 5, 1, 12, 0, 0, tzinfo=UTC)
    post_low = Post(
        source="lihkg",
        ticker="0700.HK",
        text="low engagement",
        created_at_utc=dt,
        likes=0,
        reposts=0,
        views=0,
        author_followers=0,
    )
    assert post_low.engagement_weight >= 1.0

    post_high = Post(
        source="lihkg",
        ticker="0700.HK",
        text="high engagement",
        created_at_utc=dt,
        likes=50,
        reposts=10,
        views=5000,
        author_followers=1000,
    )
    assert post_high.engagement_weight > post_low.engagement_weight


def test_sentiment_score_validation() -> None:
    score = SentimentScore(
        content_hash="abc12345",
        sentiment=0.75,
        direction=Direction.BULL,
        confidence=0.9,
        mentions_position=PositionClaim.LONG,
        language="zh",
    )
    assert score.sentiment == 0.75
    assert score.direction == Direction.BULL

    # Sentiment out of bounds must fail
    with pytest.raises(ValidationError):
        SentimentScore(
            content_hash="abc12345",
            sentiment=1.5,
            direction=Direction.BULL,
            confidence=0.9,
        )

    with pytest.raises(ValidationError):
        SentimentScore(
            content_hash="abc12345",
            sentiment=-0.5,
            direction=Direction.BEAR,
            confidence=1.2,
        )


def test_ohlcv_bar_validation() -> None:
    dt = datetime(2025, 5, 1, 9, 30, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    bar = OHLCVBar(
        ticker="0700.HK",
        timestamp_utc=dt,
        open=350.0,
        high=355.0,
        low=348.0,
        close=352.0,
        volume=1200000.0,
    )
    assert bar.timestamp_utc.tzinfo == UTC
    assert bar.close == 352.0


def test_ticker_snapshot_structure() -> None:
    snap = TickerSnapshot(
        ticker="0700.HK",
        asset_name="Tencent Holdings",
        current_price=352.0,
        weighted_sentiment=0.45,
        direction=Direction.BULL,
        mention_count=18,
        crowd_position=CrowdStance(
            long_claims=12,
            short_claims=2,
            none_claims=4,
            long_pct=66.7,
            short_pct=11.1,
            suggested_stance="Bullish consensus",
        ),
        support_resistance_levels=[
            SupportResistanceLevel(
                level_type=LevelType.SUPPORT,
                price=345.0,
                strength=2.0,
                method="argrelextrema",
            ),
            SupportResistanceLevel(
                level_type=LevelType.RESISTANCE,
                price=360.0,
                strength=2.0,
                method="argrelextrema",
            ),
        ],
        news_summary="Tencent reported strong cloud revenue growth. Gaming business steady.",
    )
    assert snap.ticker == "0700.HK"
    assert len(snap.support_resistance_levels) == 2
    assert snap.crowd_position.suggested_stance == "Bullish consensus"


def test_content_hash_consistency() -> None:
    h1 = compute_content_hash("Test text", "lihkg", "0700.HK")
    h2 = compute_content_hash("Test text", "lihkg", "0700.HK")
    h3 = compute_content_hash("Other text", "lihkg", "0700.HK")
    assert h1 == h2
    assert h1 != h3


def test_clean_architecture_domain_isolation() -> None:
    """Ensure domain modules do not import any outer layers."""
    import domain.constants
    import domain.enums
    import domain.interfaces
    import domain.models

    for mod in [domain.constants, domain.enums, domain.interfaces, domain.models]:
        for k, v in mod.__dict__.items():
            if hasattr(v, "__module__") and v.__module__:
                mod_name = v.__module__
                assert not mod_name.startswith("application"), f"Domain leaked to application: {k}"
                assert not mod_name.startswith("infrastructure"), f"Domain leaked to infrastructure: {k}"
                assert not mod_name.startswith("interfaces"), f"Domain leaked to interfaces: {k}"
                assert not mod_name.startswith("agents"), f"Domain leaked to agents: {k}"


def test_tickers_constant_valid() -> None:
    assert len(TICKERS_BY_EXCHANGE[Exchange.HKEX]) == 5
    assert len(TICKERS_BY_EXCHANGE[Exchange.KRX]) == 5
    symbols_hk = [t["symbol"] for t in TICKERS_BY_EXCHANGE[Exchange.HKEX]]
    symbols_kr = [t["symbol"] for t in TICKERS_BY_EXCHANGE[Exchange.KRX]]
    assert "0700.HK" in symbols_hk
    assert "005930.KS" in symbols_kr
