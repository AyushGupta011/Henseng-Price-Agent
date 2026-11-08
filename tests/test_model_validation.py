import numpy as np
import pandas as pd

from application.model_engine import PooledModelEngine
from domain.enums import Exchange


def test_walk_forward_validation_and_baselines() -> None:
    engine = PooledModelEngine(exchange=Exchange.HKEX, embargo_bars=1)

    # Generate synthetic pooled dataset (50 bars)
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2025-05-01", periods=n, freq="h", tz="UTC")

    records = []
    for i in range(n):
        ret = np.random.normal(0.001, 0.01)
        close = 100.0 * np.exp(np.cumsum([ret])[0])
        records.append(
            {
                "ticker": "0700.HK",
                "timestamp_utc": dates[i],
                "close": close,
                "next_close": close * 1.002,
                "target_next_return": ret,
                "return_lag_1": ret,
                "return_lag_2": 0.001,
                "return_lag_3": -0.001,
                "rolling_vol_3": 0.005,
                "rolling_vol_6": 0.007,
                "weighted_sentiment": np.random.uniform(-0.5, 0.5),
                "weighted_sentiment_lag_1": 0.1,
                "weighted_sentiment_lag_2": 0.0,
                "weighted_sentiment_lag_3": 0.0,
                "mean_sentiment": 0.1,
                "mean_sentiment_lag_1": 0.1,
                "mean_sentiment_lag_2": 0.0,
                "mean_sentiment_lag_3": 0.0,
                "sentiment_std": 0.2,
                "sentiment_std_lag_1": 0.2,
                "sentiment_std_lag_2": 0.0,
                "sentiment_std_lag_3": 0.0,
                "mention_count": 5,
                "mention_count_lag_1": 5,
                "mention_count_lag_2": 0,
                "mention_count_lag_3": 0,
                "news_sentiment": 0.0,
                "news_sentiment_lag_1": 0.0,
                "news_sentiment_lag_2": 0.0,
                "news_sentiment_lag_3": 0.0,
            }
        )

    df = pd.DataFrame(records)
    metrics, eval_rows = engine.walk_forward_validate(
        df, model_type="ridge", min_train_size=20, test_step=5
    )

    assert metrics.sample_count > 0
    assert metrics.directional_accuracy >= 0.0
    assert metrics.mae >= 0.0
    assert metrics.baseline_price_only_mae >= 0.0
    assert 0.0 <= metrics.binomial_p_value <= 1.0
    assert len(eval_rows) == metrics.sample_count
    assert "Walk-forward validated" in metrics.notes
