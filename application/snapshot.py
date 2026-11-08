import logging
from datetime import UTC, datetime

from application.sentiment import ScoredItem
from domain.enums import Direction, PositionClaim
from domain.models import (
    CrowdStance,
    NewsArticle,
    OHLCVBar,
    SupportResistanceLevel,
    Ticker,
    TickerSnapshot,
)
from infrastructure.analysis.support_resistance import SupportResistanceAnalyzer

logger = logging.getLogger(__name__)


class SnapshotBuilderUseCase:
    """Builds comprehensive per-ticker snapshots including sentiment, crowd stance, and S/R levels."""

    def __init__(self, sr_analyzer: SupportResistanceAnalyzer | None = None) -> None:
        self.sr_analyzer = sr_analyzer or SupportResistanceAnalyzer()

    def build_snapshot(
        self,
        ticker: Ticker,
        bars: list[OHLCVBar],
        scored_items: list[ScoredItem],
        news_articles: list[NewsArticle],
    ) -> TickerSnapshot:
        # Current price
        current_price = bars[-1].close if bars else 0.0

        # Filter items for this ticker
        ticker_items = [it for it in scored_items if it.ticker == ticker.symbol]
        mention_count = len(ticker_items)

        # Weighted sentiment
        if ticker_items:
            total_weight = sum(it.engagement_weight for it in ticker_items)
            weighted_sum = sum(it.sentiment * it.engagement_weight for it in ticker_items)
            weighted_sentiment = weighted_sum / (total_weight + 1e-9)
        else:
            weighted_sentiment = 0.0

        # Direction
        if weighted_sentiment > 0.15:
            overall_direction = Direction.BULL
        elif weighted_sentiment < -0.15:
            overall_direction = Direction.BEAR
        else:
            overall_direction = Direction.NEUTRAL

        # Crowd stance from explicit claims
        long_claims = sum(1 for it in ticker_items if it.mentions_position == PositionClaim.LONG.value)
        short_claims = sum(1 for it in ticker_items if it.mentions_position == PositionClaim.SHORT.value)
        none_claims = sum(1 for it in ticker_items if it.mentions_position == PositionClaim.NONE.value)
        total_claims = long_claims + short_claims

        if total_claims > 0:
            long_pct = round((long_claims / total_claims) * 100.0, 1)
            short_pct = round((short_claims / total_claims) * 100.0, 1)
            if long_pct >= 65.0:
                suggested_stance = "Bullish Consensus (Crowd Net Long)"
            elif short_pct >= 65.0:
                suggested_stance = "Bearish Consensus (Crowd Net Short)"
            else:
                suggested_stance = "Split / Contested Opinion"
        else:
            long_pct = 0.0
            short_pct = 0.0
            suggested_stance = "Neutral / Low retail participation"

        crowd_stance = CrowdStance(
            long_claims=long_claims,
            short_claims=short_claims,
            none_claims=none_claims,
            long_pct=long_pct,
            short_pct=short_pct,
            suggested_stance=suggested_stance,
        )

        # Support & Resistance levels
        sr_levels: list[SupportResistanceLevel] = []
        if bars:
            sr_levels = self.sr_analyzer.compute_levels(bars)

        # 2-line news summary
        ticker_news = [n for n in news_articles if n.ticker == ticker.symbol]
        news_summary = self._summarize_news(ticker_news, ticker)

        return TickerSnapshot(
            ticker=ticker.symbol,
            asset_name=f"{ticker.name_en} ({ticker.name_local})",
            current_price=round(current_price, 2),
            weighted_sentiment=round(weighted_sentiment, 3),
            direction=overall_direction,
            mention_count=mention_count,
            crowd_position=crowd_stance,
            support_resistance_levels=sr_levels,
            news_summary=news_summary,
            as_of_utc=datetime.now(UTC),
        )

    def _summarize_news(self, news: list[NewsArticle], ticker: Ticker) -> str:
        if not news:
            return f"1. Recent market trading steady for {ticker.name_en}.\n2. No major corporate disclosures in the last 10 days."

        # Take the top 2 titles/snippets
        lines: list[str] = []
        for i, article in enumerate(news[:2], 1):
            snippet = article.title
            if article.text_snippet and len(article.text_snippet) > 10:
                short_body = article.text_snippet[:90].replace("\n", " ").strip()
                lines.append(f"{i}. {snippet} — {short_body}...")
            else:
                lines.append(f"{i}. {snippet}")

        if len(lines) == 1:
            lines.append("2. Monitoring follow-up analyst consensus and volume shifts.")

        return "\n".join(lines)
