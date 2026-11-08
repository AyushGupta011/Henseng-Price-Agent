import logging
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf

from domain.models import OHLCVBar

logger = logging.getLogger(__name__)


class YFinanceMarketSource:
    """Fetches 1h OHLCV bars via yfinance with timezone-aware UTC timestamps."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def fetch_bars(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
        days: int = 14,
        interval: str = "1h",
    ) -> list[OHLCVBar]:
        """Fetch 1h OHLCV bars, converting all bar times to UTC."""
        if end is None:
            end = datetime.now(UTC)
        if start is None:
            start = end - timedelta(days=days)

        start_str = start.strftime("%Y-%m-%d")
        # Add 1 day to end_str for yfinance exclusive boundary
        end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                yf_ticker = yf.Ticker(ticker)
                df: pd.DataFrame = yf_ticker.history(
                    start=start_str,
                    end=end_str,
                    interval=interval,
                    auto_adjust=True,
                )
                if df.empty:
                    # Fallback to period='1mo' if date range yielded 0 bars
                    df = yf_ticker.history(period="1mo", interval=interval, auto_adjust=True)

                if df.empty:
                    logger.warning("No yfinance bars returned for %s", ticker)
                    return []

                bars: list[OHLCVBar] = []
                for idx, row in df.iterrows():
                    # Handle index timestamp
                    if isinstance(idx, pd.Timestamp):
                        ts: pd.Timestamp = idx
                    else:
                        ts = pd.to_datetime(idx)

                    if ts.tzinfo is None:
                        # Default to UTC if naive
                        ts_utc = ts.tz_localize(UTC)
                    else:
                        ts_utc = ts.tz_convert(UTC)

                    py_dt: datetime = ts_utc.to_pydatetime()
                    bars.append(
                        OHLCVBar(
                            ticker=ticker,
                            timestamp_utc=py_dt,
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=float(row.get("Volume", 0.0)),
                        )
                    )

                bars.sort(key=lambda b: b.timestamp_utc)
                return bars
            except Exception as e:
                last_exc = e
                wait_time = self.backoff_factor * (2**attempt)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                    attempt + 1,
                    self.max_retries,
                    ticker,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)

        logger.error("Failed to fetch bars for %s after %d retries: %s", ticker, self.max_retries, last_exc)
        return []
