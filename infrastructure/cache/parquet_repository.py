from datetime import datetime
from pathlib import Path

import pandas as pd

from domain.models import NewsArticle, OHLCVBar, Post


class ParquetRawRepository:
    """Parquet cache repository for raw data in data/raw/ directory."""

    def __init__(self, raw_dir: Path | str = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _posts_path(self, exchange: str, source: str) -> Path:
        return self.raw_dir / f"posts_{exchange.lower()}_{source.lower()}.parquet"

    def _news_path(self, exchange: str) -> Path:
        return self.raw_dir / f"news_{exchange.lower()}.parquet"

    def _bars_path(self, ticker: str) -> Path:
        sanitized = ticker.replace(".", "_").lower()
        return self.raw_dir / f"bars_{sanitized}.parquet"

    def save_posts(self, posts: list[Post], exchange: str, source: str) -> None:
        if not posts:
            return
        path = self._posts_path(exchange, source)
        records = [
            {
                "source": p.source,
                "ticker": p.ticker,
                "text": p.text,
                "created_at_utc": p.created_at_utc.isoformat(),
                "likes": p.likes,
                "reposts": p.reposts,
                "views": p.views,
                "author_followers": p.author_followers,
                "url": p.url,
                "content_hash": p.content_hash,
            }
            for p in posts
        ]
        df = pd.DataFrame(records)

        # Merge with existing cache if present
        if path.exists():
            try:
                existing_df = pd.read_parquet(path)
                df = (
                    pd.concat([existing_df, df], ignore_index=True)
                    .drop_duplicates(subset=["content_hash"])
                    .reset_index(drop=True)
                )
            except Exception:
                pass

        df.to_parquet(path, index=False, engine="pyarrow")

    def load_posts(
        self, exchange: str, source: str, ticker: str | None = None
    ) -> list[Post]:
        path = self._posts_path(exchange, source)
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(path)
            if ticker is not None and not df.empty:
                df = df[df["ticker"] == ticker]
            posts: list[Post] = []
            for _, row in df.iterrows():
                dt = datetime.fromisoformat(str(row["created_at_utc"]))
                posts.append(
                    Post(
                        source=str(row["source"]),
                        ticker=str(row["ticker"]),
                        text=str(row["text"]),
                        created_at_utc=dt,
                        likes=int(row.get("likes", 0)),
                        reposts=int(row.get("reposts", 0)),
                        views=int(row.get("views", 0)),
                        author_followers=int(row.get("author_followers", 0)),
                        url=str(row.get("url", "")),
                        content_hash=str(row.get("content_hash", "")),
                    )
                )
            return posts
        except Exception:
            return []

    def save_news(self, news: list[NewsArticle], exchange: str) -> None:
        if not news:
            return
        path = self._news_path(exchange)
        records = [
            {
                "source": n.source,
                "ticker": n.ticker,
                "title": n.title,
                "url": n.url,
                "published_at_utc": n.published_at_utc.isoformat(),
                "text_snippet": n.text_snippet,
                "content_hash": n.content_hash,
            }
            for n in news
        ]
        df = pd.DataFrame(records)
        if path.exists():
            try:
                existing_df = pd.read_parquet(path)
                df = (
                    pd.concat([existing_df, df], ignore_index=True)
                    .drop_duplicates(subset=["content_hash"])
                    .reset_index(drop=True)
                )
            except Exception:
                pass
        df.to_parquet(path, index=False, engine="pyarrow")

    def load_news(self, exchange: str, ticker: str | None = None) -> list[NewsArticle]:
        path = self._news_path(exchange)
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(path)
            if ticker is not None and not df.empty:
                df = df[df["ticker"] == ticker]
            articles: list[NewsArticle] = []
            for _, row in df.iterrows():
                dt = datetime.fromisoformat(str(row["published_at_utc"]))
                articles.append(
                    NewsArticle(
                        source=str(row["source"]),
                        ticker=str(row["ticker"]),
                        title=str(row["title"]),
                        url=str(row.get("url", "")),
                        published_at_utc=dt,
                        text_snippet=str(row.get("text_snippet", "")),
                        content_hash=str(row.get("content_hash", "")),
                    )
                )
            return articles
        except Exception:
            return []

    def save_bars(self, bars: list[OHLCVBar], ticker: str) -> None:
        if not bars:
            return
        path = self._bars_path(ticker)
        records = [
            {
                "ticker": b.ticker,
                "timestamp_utc": b.timestamp_utc.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        df = pd.DataFrame(records)
        if path.exists():
            try:
                existing_df = pd.read_parquet(path)
                df = (
                    pd.concat([existing_df, df], ignore_index=True)
                    .drop_duplicates(subset=["timestamp_utc"])
                    .sort_values(by="timestamp_utc")
                    .reset_index(drop=True)
                )
            except Exception:
                pass
        df.to_parquet(path, index=False, engine="pyarrow")

    def load_bars(self, ticker: str) -> list[OHLCVBar]:
        path = self._bars_path(ticker)
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(path)
            bars: list[OHLCVBar] = []
            for _, row in df.iterrows():
                dt = datetime.fromisoformat(str(row["timestamp_utc"]))
                bars.append(
                    OHLCVBar(
                        ticker=str(row["ticker"]),
                        timestamp_utc=dt,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
            bars.sort(key=lambda b: b.timestamp_utc)
            return bars
        except Exception:
            return []
