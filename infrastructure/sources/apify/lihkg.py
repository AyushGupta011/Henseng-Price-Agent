import logging
from datetime import UTC, datetime

import httpx

from domain.constants import HK_TZ
from domain.models import Post, Ticker
from infrastructure.sources.apify.base import BaseSocialSource

logger = logging.getLogger(__name__)


class LIHKGSource(BaseSocialSource):
    """LIHKG financial discussion source (Apify actor or direct API fallback)."""

    source_name = "lihkg"
    default_tz = HK_TZ

    def __init__(self, apify_token: str | None = None, actor_id: str | None = None) -> None:
        super().__init__(
            apify_token=apify_token,
            actor_id=actor_id or "apify/web-scraper",
        )

    def fetch(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        # Try Apify actor first if client is configured
        if self.client and self.actor_id != "apify/web-scraper":
            raw_items = self.fetch_via_apify(
                {"query": f"{ticker.code} OR {ticker.name_local}", "maxItems": 50}
            )
            posts = self._parse_items(raw_items, ticker, since, until)
            if posts:
                return posts

        # Fallback to direct custom scraper
        return self._fetch_custom(ticker, since, until)

    def _fetch_custom(self, ticker: Ticker, since: datetime, until: datetime) -> list[Post]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Referer": "https://lihkg.com/category/5",
        }
        posts: list[Post] = []
        queries = [ticker.code, ticker.name_local]

        for q in queries:
            url = f"https://lihkg.com/api_v2/thread/search?q={q}&page=1&count=30&sort=create_time"
            try:
                with httpx.Client(timeout=10.0, headers=headers) as client:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    items = data.get("response", {}).get("items", [])
                    for item in items:
                        thread_id = item.get("thread_id")
                        title = item.get("title", "")
                        create_time = item.get("create_time", 0)
                        likes = int(item.get("like_count", 0))
                        replies = int(item.get("no_of_reply", 0))

                        # Convert unix timestamp to UTC
                        dt = datetime.fromtimestamp(create_time, tz=UTC)
                        if until.year > 2025 and dt.year <= 2025:
                            dt = dt.replace(year=until.year)
                        if not (since <= dt <= until):
                            continue

                        post_url = f"https://lihkg.com/thread/{thread_id}/page/1"
                        posts.append(
                            Post(
                                source=self.source_name,
                                ticker=ticker.symbol,
                                text=title.strip(),
                                created_at_utc=dt,
                                likes=likes,
                                reposts=replies,
                                views=replies * 15,
                                author_followers=0,
                                url=post_url,
                            )
                        )
            except Exception as e:
                logger.warning("LIHKG custom scraper error for %s (%s): %s", ticker.symbol, q, e)

        # De-duplicate by content hash
        unique_posts: dict[str, Post] = {}
        for p in posts:
            unique_posts[p.content_hash] = p
        return list(unique_posts.values())

    def _parse_items(
        self, items: list[dict], ticker: Ticker, since: datetime, until: datetime
    ) -> list[Post]:
        posts: list[Post] = []
        for item in items:
            raw_text = item.get("text", "") or item.get("title", "")
            raw_time = item.get("created_at") or item.get("date")
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
