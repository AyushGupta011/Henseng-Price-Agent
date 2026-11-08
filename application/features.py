import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from application.sentiment import ScoredItem
from domain.models import OHLCVBar, Ticker

logger = logging.getLogger(__name__)


def compute_leakage_free_features(
    bars: list[OHLCVBar],
    scored_items: list[ScoredItem],
    ticker: Ticker,
    lookback_hours: int = 4,
) -> pd.DataFrame:
    """
    Constructs ML features with strict forward-looking leakage prevention.
    Rule: A post/news published at time t may ONLY influence bars whose START TIME is >= t.
    """
    if len(bars) < 10:
        return pd.DataFrame()

    sorted_bars = sorted(bars, key=lambda b: b.timestamp_utc)
    records: list[dict[str, object]] = []

    for i in range(len(sorted_bars)):
        bar = sorted_bars[i]
        bar_start_utc = bar.timestamp_utc

        # Filter items strictly published BEFORE or AT bar_start_utc
        # AND within lookback_hours prior to bar_start_utc
        window_start = bar_start_utc - timedelta(hours=lookback_hours)

        eligible_items = [
            it
            for it in scored_items
            if it.ticker == ticker.symbol and (window_start <= it.timestamp_utc <= bar_start_utc)
        ]

        social_items = [it for it in eligible_items if not it.is_news]
        news_items = [it for it in eligible_items if it.is_news]

        # Calculate sentiment metrics in window
        if social_items:
            weights = np.array([it.engagement_weight for it in social_items], dtype=float)
            sents = np.array([it.sentiment for it in social_items], dtype=float)
            w_sum = np.sum(weights)
            weighted_sent = float(np.sum(sents * weights) / (w_sum + 1e-9))
            mean_sent = float(np.mean(sents))
            std_sent = float(np.std(sents)) if len(sents) > 1 else 0.0
            mention_cnt = len(social_items)
        else:
            weighted_sent = 0.0
            mean_sent = 0.0
            std_sent = 0.0
            mention_cnt = 0

        news_sent = float(np.mean([it.sentiment for it in news_items])) if news_items else 0.0

        records.append(
            {
                "ticker": ticker.symbol,
                "timestamp_utc": bar_start_utc,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "weighted_sentiment": weighted_sent,
                "mean_sentiment": mean_sent,
                "sentiment_std": std_sent,
                "mention_count": mention_cnt,
                "news_sentiment": news_sent,
            }
        )

    df = pd.DataFrame(records)
    if df.empty or len(df) < 5:
        return pd.DataFrame()

    # Price returns: log return of current bar close vs previous close
    df["log_close"] = np.log(df["close"] + 1e-9)
    df["return_lag_1"] = df["log_close"] - df["log_close"].shift(1)
    df["return_lag_2"] = df["log_close"].shift(1) - df["log_close"].shift(2)
    df["return_lag_3"] = df["log_close"].shift(2) - df["log_close"].shift(3)

    # Rolling volatility of returns (3 and 6 periods)
    df["rolling_vol_3"] = df["return_lag_1"].rolling(3).std().fillna(0.0)
    df["rolling_vol_6"] = df["return_lag_1"].rolling(6).std().fillna(0.0)

    # Lags for sentiment features (1, 2, 3)
    for feat in ["weighted_sentiment", "mean_sentiment", "sentiment_std", "mention_count", "news_sentiment"]:
        df[f"{feat}_lag_1"] = df[feat].shift(1).fillna(0.0)
        df[f"{feat}_lag_2"] = df[feat].shift(2).fillna(0.0)
        df[f"{feat}_lag_3"] = df[feat].shift(3).fillna(0.0)

    # Target: NEXT-BAR log return ln(Close_{t+1} / Close_t)
    df["target_next_return"] = df["log_close"].shift(-1) - df["log_close"]
    df["next_close"] = df["close"].shift(-1)

    # Drop NaNs created by lagging (first 3 rows)
    df = df.iloc[3:].reset_index(drop=True)
    return df
