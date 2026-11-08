import hashlib
import math
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain.enums import Direction, Exchange, LevelType, PositionClaim


def compute_content_hash(text: str, source: str = "", identifier: str = "") -> str:
    raw = f"{source}::{identifier}::{text}".strip().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class Ticker(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    exchange: Exchange
    name_en: str
    name_local: str
    code: str
    search_keywords: list[str] = Field(default_factory=list)


class Post(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    ticker: str
    text: str
    created_at_utc: datetime
    likes: int = 0
    reposts: int = 0
    views: int = 0
    author_followers: int = 0
    url: str = ""
    content_hash: str = ""

    @field_validator("created_at_utc", mode="after")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware UTC")
        return v.astimezone(UTC)

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            computed = compute_content_hash(
                self.text, self.source, f"{self.ticker}::{self.created_at_utc.isoformat()}"
            )
            object.__setattr__(self, "content_hash", computed)

    @property
    def engagement_weight(self) -> float:
        """
        Engagement weight = log1p(likes + reposts + views / 100) * log1p(followers)
        Bounded safely to avoid zeros when engagement is minimal.
        """
        raw_inter = max(0.0, float(self.likes + self.reposts + (self.views / 100.0)))
        followers = max(0.0, float(self.author_followers))
        # Ensure a base weight of 1.0 so unliked posts still count reasonably
        weight = math.log1p(raw_inter) * math.log1p(followers)
        return max(1.0, weight)


class NewsArticle(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    ticker: str
    title: str
    url: str
    published_at_utc: datetime
    text_snippet: str
    content_hash: str = ""

    @field_validator("published_at_utc", mode="after")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("published_at_utc must be timezone-aware UTC")
        return v.astimezone(UTC)

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            computed = compute_content_hash(
                f"{self.title} {self.text_snippet}",
                self.source,
                f"{self.ticker}::{self.url}",
            )
            object.__setattr__(self, "content_hash", computed)


class SentimentScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    content_hash: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    mentions_position: PositionClaim = PositionClaim.NONE
    language: str = "en"
    raw_response: str = ""
    engagement_weight: float = 1.0


class OHLCVBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("timestamp_utc", mode="after")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware UTC")
        return v.astimezone(UTC)


class SupportResistanceLevel(BaseModel):
    model_config = ConfigDict(frozen=True)

    level_type: LevelType
    price: float
    strength: float = 1.0
    method: str = "argrelextrema"


class CrowdStance(BaseModel):
    model_config = ConfigDict(frozen=True)

    long_claims: int = 0
    short_claims: int = 0
    none_claims: int = 0
    long_pct: float = 0.0
    short_pct: float = 0.0
    suggested_stance: str = "Neutral / Low consensus"


class TickerSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    asset_name: str
    current_price: float
    weighted_sentiment: float
    direction: Direction
    mention_count: int
    crowd_position: CrowdStance
    support_resistance_levels: list[SupportResistanceLevel] = Field(default_factory=list)
    news_summary: str = ""
    as_of_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PredictionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    target_bar_time: datetime
    current_price: float
    predicted_log_return: float
    predicted_price: float
    actual_log_return: float | None = None
    actual_price: float | None = None
    model_name: str = ""
    features_used: dict[str, float] = Field(default_factory=dict)


class ValidationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    exchange: Exchange
    sample_count: int
    mae: float
    rmse: float
    directional_accuracy: float
    binomial_p_value: float
    baseline_no_change_mae: float
    baseline_last_direction_acc: float
    baseline_price_only_mae: float
    baseline_price_only_acc: float
    lift_mae: float
    lift_directional_acc: float
    sentiment_adds_lift: bool
    notes: str = ""
