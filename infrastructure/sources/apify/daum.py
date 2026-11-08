import logging
from datetime import UTC, datetime

import httpx

from domain.constants import KR_TZ
from domain.models import Post, Ticker
from infrastructure.sources.apify.base import BaseSocialSource

logger = logging.getLogger(__name__)


class DaumFinanceSource(BaseSocialSource):
    """Daum Finance Discussion scraper."""

    source_name = "daum_finance"
    default_tz = KR_TZ

    def __init__(self, apify_token: str | None = None, actor_id: str | None = None) -> None:
        super().__init__(
            apify_token=apify_token,
            actor_id=actor_id or "apify/web-scraper",
        )

    def fetch(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        if self.client and self.actor_id != "apify/web-scraper":
            raw_items = self.fetch_via_apify(
                {"code": ticker.code, "source": "daum", "maxItems": 40}
            )
            if raw_items:
                return self._parse_items(raw_items, ticker, since, until)

        return self._fetch_custom(ticker, since, until)

    def _fetch_custom(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        symbol_code = f"A{ticker.code}"
        url = f"https://finance.daum.net/api/talks/posts?limit=30&symbolCode={symbol_code}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": f"https://finance.daum.net/quotes/{symbol_code}",
            "Accept": "application/json",
        }
        posts: list[Post] = []
        try:
            with httpx.Client(timeout=10.0, headers=headers) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", [])
                    for item in items:
                        title = item.get("title", "") or item.get("content", "")
                        created_at_str = item.get("createdAt", "")
                        post_id = item.get("id")

                        # Parse ISO datetime
                        post_dt = until
                        if created_at_str:
                            try:
                                parsed = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                                post_dt = parsed.astimezone(UTC)
                            except Exception:
                                post_dt = until

                        if not (since <= post_dt <= until):
                            continue

                        view_count = int(item.get("viewCount", 0))
                        like_count = int(item.get("likeCount", 0))
                        comment_count = int(item.get("commentCount", 0))
                        post_url = f"https://finance.daum.net/talks/{post_id}"

                        posts.append(
                            Post(
                                source=self.source_name,
                                ticker=ticker.symbol,
                                text=title.strip(),
                                created_at_utc=post_dt,
                                likes=like_count,
                                reposts=comment_count,
                                views=view_count,
                                author_followers=0,
                                url=post_url,
                            )
                        )
        except Exception as e:
            logger.warning("Daum Finance custom scraper failed for %s: %s", ticker.symbol, e)

        unique: dict[str, Post] = {}
        for p in posts:
            unique[p.content_hash] = p
        return list(unique.values())

    def _parse_items(
        self, items: list[dict], ticker: Ticker, since: datetime, until: datetime
    ) -> list[Post]:
        posts: list[Post] = []
        for item in items:
            raw_text = item.get("title", "") or item.get("text", "")
            raw_time = item.get("date") or item.get("created_at")
            try:
                dt = (
                    datetime.fromisoformat(str(raw_time)).astimezone(UTC)
                    if raw_time
                    else datetime.now(UTC)
                )
            except Exception:
                dt = datetime.now(UTC)

            if not (since <= dt <= until):
                continue

            posts.append(
                Post(
                    source=self.source_name,
                    ticker=ticker.symbol,
                    text=raw_text.strip(),
                    created_at_utc=dt,
                    likes=int(item.get("likes", 0)),
                    reposts=int(item.get("reposts", 0)),
                    views=int(item.get("views", 0)),
                    author_followers=int(item.get("author_followers", 0)),
                    url=str(item.get("url", "")),
                )
            )
        return posts
