import logging
from datetime import datetime
from typing import Any

from domain.models import NewsArticle, Post
from infrastructure.llm.sentiment_scorer import LLMSentimentScorer

logger = logging.getLogger(__name__)


class ScoredItem:

    def __init__(
        self,
        ticker: str,
        timestamp_utc: datetime,
        sentiment: float,
        direction: str,
        confidence: float,
        engagement_weight: float,
        mentions_position: str,
        language: str,
        is_news: bool = False,
    ) -> None:
        self.ticker = ticker
        self.timestamp_utc = timestamp_utc
        self.sentiment = sentiment
        self.direction = direction
        self.confidence = confidence
        self.engagement_weight = engagement_weight
        self.mentions_position = mentions_position
        self.language = language
        self.is_news = is_news


class SentimentAnalysisUseCase:
    """Orchestrates sentiment scoring across posts and news articles."""

    def __init__(self, scorer: LLMSentimentScorer | None = None) -> None:
        self.scorer = scorer or LLMSentimentScorer()

    def execute(
        self,
        posts: list[Post],
        news: list[NewsArticle],
    ) -> list[ScoredItem]:
        logger.info(
            "Scoring sentiment for %d social posts and %d news articles...",
            len(posts),
            len(news),
        )

        items_to_score: list[dict[str, Any]] = []

        for p in posts:
            items_to_score.append(
                {
                    "text": p.text,
                    "content_hash": p.content_hash,
                    "engagement_weight": p.engagement_weight,
                    "ticker": p.ticker,
                    "timestamp_utc": p.created_at_utc,
                    "is_news": False,
                }
            )

        for n in news:
            items_to_score.append(
                {
                    "text": f"{n.title}. {n.text_snippet}",
                    "content_hash": n.content_hash,
                    "engagement_weight": 2.5,  # News articles have standard baseline weight
                    "ticker": n.ticker,
                    "timestamp_utc": n.published_at_utc,
                    "is_news": True,
                }
            )

        # Score in batch with hashing cache
        sentiment_scores = self.scorer.score_batch(items_to_score)

        scored_results: list[ScoredItem] = []
        for raw, score in zip(items_to_score, sentiment_scores, strict=False):
            scored_results.append(
                ScoredItem(
                    ticker=raw["ticker"],
                    timestamp_utc=raw["timestamp_utc"],
                    sentiment=score.sentiment,
                    direction=score.direction.value,
                    confidence=score.confidence,
                    engagement_weight=score.engagement_weight,
                    mentions_position=score.mentions_position.value,
                    language=score.language,
                    is_news=raw["is_news"],
                )
            )

        return scored_results
