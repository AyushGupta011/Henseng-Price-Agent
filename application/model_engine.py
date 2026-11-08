import logging

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import Ridge

from domain.enums import Exchange
from domain.models import PredictionResult, ValidationMetrics

pd.set_option("future.no_silent_downcasting", True)
logger = logging.getLogger(__name__)

FEATURE_COLS_SENTIMENT = [
    "return_lag_1", "return_lag_2", "return_lag_3",
    "rolling_vol_3", "rolling_vol_6",
    "weighted_sentiment", "weighted_sentiment_lag_1", "weighted_sentiment_lag_2", "weighted_sentiment_lag_3",
    "mean_sentiment", "mean_sentiment_lag_1", "mean_sentiment_lag_2", "mean_sentiment_lag_3",
    "sentiment_std", "sentiment_std_lag_1", "sentiment_std_lag_2", "sentiment_std_lag_3",
    "mention_count", "mention_count_lag_1", "mention_count_lag_2", "mention_count_lag_3",
    "news_sentiment", "news_sentiment_lag_1", "news_sentiment_lag_2", "news_sentiment_lag_3",
]

FEATURE_COLS_PRICE_ONLY = [
    "return_lag_1", "return_lag_2", "return_lag_3",
    "rolling_vol_3", "rolling_vol_6",
]


class PooledModelEngine:
    """Trains pooled models with ticker effects using walk-forward validation and honest baselines."""

    def __init__(self, exchange: Exchange, embargo_bars: int = 1) -> None:
        self.exchange = exchange
        self.embargo_bars = embargo_bars

    def prepare_pooled_dataset(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        # Sort chronologically by timestamp to preserve walk-forward order
        combined = combined.sort_values(by="timestamp_utc").reset_index(drop=True)
        # Add ticker fixed effects (one-hot encoding)
        ticker_dummies = pd.get_dummies(combined["ticker"], prefix="ticker", drop_first=True, dtype=float)
        return pd.concat([combined, ticker_dummies], axis=1)

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        model_type: str = "ridge",
        min_train_size: int = 25,
        test_step: int = 5,
    ) -> tuple[ValidationMetrics, list[dict[str, object]]]:
        # Filter rows where target is present (drop the final bar which has no next return)
        valid_df = df.dropna(subset=["target_next_return"]).copy().reset_index(drop=True)
        n = len(valid_df)

        if n < min_train_size + self.embargo_bars + 3:
            logger.warning("Insufficient samples (%d) for walk-forward validation", n)
            return (
                ValidationMetrics(
                    model_name=model_type,
                    exchange=self.exchange,
                    sample_count=n,
                    mae=0.0,
                    rmse=0.0,
                    directional_accuracy=50.0,
                    binomial_p_value=1.0,
                    baseline_no_change_mae=0.0,
                    baseline_last_direction_acc=50.0,
                    baseline_price_only_mae=0.0,
                    baseline_price_only_acc=50.0,
                    lift_mae=0.0,
                    lift_directional_acc=0.0,
                    sentiment_adds_lift=False,
                    notes="Insufficient history for out-of-sample walk-forward validation.",
                ),
                [],
            )

        ticker_cols = [c for c in df.columns if c.startswith("ticker_")]
        full_feature_cols = [c for c in FEATURE_COLS_SENTIMENT if c in df.columns] + ticker_cols
        price_feature_cols = [c for c in FEATURE_COLS_PRICE_ONLY if c in df.columns] + ticker_cols

        preds_full: list[float] = []
        preds_price_only: list[float] = []
        actuals: list[float] = []
        last_dir_preds: list[float] = []
        rows_eval: list[dict[str, object]] = []

        # Expanding window loop with embargo gap
        current_train_end = min_train_size
        while current_train_end + self.embargo_bars < n:
            train_idx = list(range(0, current_train_end))
            test_start = current_train_end + self.embargo_bars
            test_end = min(n, test_start + test_step)
            test_idx = list(range(test_start, test_end))

            train_data = valid_df.iloc[train_idx]
            test_data = valid_df.iloc[test_idx]

            X_train = train_data[full_feature_cols].fillna(0.0).values
            y_train = train_data["target_next_return"].values

            X_test = test_data[full_feature_cols].fillna(0.0).values
            y_test = test_data["target_next_return"].values

            # Fit sentiment model
            if model_type == "lightgbm":
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(
                        max_depth=3,
                        n_estimators=30,
                        learning_rate=0.05,
                        reg_alpha=1.0,
                        reg_lambda=1.0,
                        random_state=42,
                        verbosity=-1,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                except Exception:
                    ridge = Ridge(alpha=10.0)
                    ridge.fit(X_train, y_train)
                    y_pred = ridge.predict(X_test)
            else:
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_train, y_train)
                y_pred = ridge.predict(X_test)

            # Fit price-only baseline model
            X_train_p = train_data[price_feature_cols].fillna(0.0).values
            X_test_p = test_data[price_feature_cols].fillna(0.0).values
            ridge_p = Ridge(alpha=10.0)
            ridge_p.fit(X_train_p, y_train)
            y_pred_p = ridge_p.predict(X_test_p)

            for i, idx_val in enumerate(test_idx):
                row = test_data.iloc[i]
                pred_val = float(y_pred[i])
                pred_p_val = float(y_pred_p[i])
                actual_val = float(y_test[i])
                last_ret = float(row.get("return_lag_1", 0.0))

                preds_full.append(pred_val)
                preds_price_only.append(pred_p_val)
                actuals.append(actual_val)
                last_dir_preds.append(last_ret)

                rows_eval.append(
                    {
                        "ticker": row["ticker"],
                        "timestamp_utc": row["timestamp_utc"],
                        "current_close": row["close"],
                        "actual_next_close": row["next_close"],
                        "predicted_return": pred_val,
                        "actual_return": actual_val,
                        "predicted_close": row["close"] * np.exp(pred_val),
                    }
                )

            current_train_end = test_end

        # Calculate metrics
        y_true = np.array(actuals)
        y_pred_arr = np.array(preds_full)
        y_pred_p_arr = np.array(preds_price_only)
        y_last_dir = np.array(last_dir_preds)

        mae = float(np.mean(np.abs(y_true - y_pred_arr)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred_arr) ** 2)))

        # Directional accuracy
        correct_directions = np.sum(np.sign(y_pred_arr) == np.sign(y_true))
        dir_acc = float((correct_directions / len(y_true)) * 100.0)

        # Baselines
        no_change_mae = float(np.mean(np.abs(y_true)))
        last_dir_correct = np.sum(np.sign(y_last_dir) == np.sign(y_true))
        last_dir_acc = float((last_dir_correct / len(y_true)) * 100.0)

        price_only_mae = float(np.mean(np.abs(y_true - y_pred_p_arr)))
        price_only_correct = np.sum(np.sign(y_pred_p_arr) == np.sign(y_true))
        price_only_acc = float((price_only_correct / len(y_true)) * 100.0)

        # Binomial test p-value: is hit rate significantly > 50% random coin toss?
        b_test = binomtest(k=int(correct_directions), n=len(y_true), p=0.5, alternative="greater")
        p_val = float(b_test.pvalue)

        lift_mae = round(price_only_mae - mae, 6)
        lift_acc = round(dir_acc - price_only_acc, 2)
        sentiment_adds_lift = bool(lift_acc > 0.0 and p_val < 0.15)

        notes = (
            f"Walk-forward validated on {len(y_true)} out-of-sample bars. "
            + (
                f"Sentiment added +{lift_acc:.1f}% directional lift over price-only baseline."
                if lift_acc > 0
                else f"Honest result: Sentiment showed no directional edge ({lift_acc:.1f}% lift vs price-only)."
            )
        )

        metrics = ValidationMetrics(
            model_name=model_type,
            exchange=self.exchange,
            sample_count=len(y_true),
            mae=round(mae, 6),
            rmse=round(rmse, 6),
            directional_accuracy=round(dir_acc, 2),
            binomial_p_value=round(p_val, 4),
            baseline_no_change_mae=round(no_change_mae, 6),
            baseline_last_direction_acc=round(last_dir_acc, 2),
            baseline_price_only_mae=round(price_only_mae, 6),
            baseline_price_only_acc=round(price_only_acc, 2),
            lift_mae=lift_mae,
            lift_directional_acc=lift_acc,
            sentiment_adds_lift=sentiment_adds_lift,
            notes=notes,
        )

        return metrics, rows_eval

    def predict_next_bar(
        self,
        df: pd.DataFrame,
        ticker_symbol: str,
        model_type: str = "ridge",
    ) -> PredictionResult:
        """Trains on all available data and predicts n+1 return and target price for the ticker."""
        ticker_df = df[df["ticker"] == ticker_symbol].copy()
        if ticker_df.empty:
            return PredictionResult(
                ticker=ticker_symbol,
                target_bar_time=pd.Timestamp.now(tz="UTC").to_pydatetime(),
                current_price=0.0,
                predicted_log_return=0.0,
                predicted_price=0.0,
                model_name=model_type,
            )

        last_row = ticker_df.iloc[-1]
        current_price = float(last_row["close"])
        bar_time = last_row["timestamp_utc"]

        ticker_cols = [c for c in df.columns if c.startswith("ticker_")]
        full_feature_cols = [c for c in FEATURE_COLS_SENTIMENT if c in df.columns] + ticker_cols

        # Train on all completed bars
        train_df = df.dropna(subset=["target_next_return"])
        if len(train_df) >= 15:
            X_train = train_df[full_feature_cols].fillna(0.0).values
            y_train = train_df["target_next_return"].values
            X_infer = last_row[full_feature_cols].fillna(0.0).values.reshape(1, -1)

            ridge = Ridge(alpha=10.0)
            ridge.fit(X_train, y_train)
            pred_return = float(ridge.predict(X_infer)[0])
        else:
            # Fallback to momentum / sentiment heuristic
            w_sent = float(last_row.get("weighted_sentiment", 0.0))
            pred_return = round(w_sent * 0.002, 5)

        pred_price = round(current_price * np.exp(pred_return), 2)

        return PredictionResult(
            ticker=ticker_symbol,
            target_bar_time=bar_time,
            current_price=round(current_price, 2),
            predicted_log_return=round(pred_return, 5),
            predicted_price=pred_price,
            model_name=model_type,
            features_used={
                "weighted_sentiment": float(last_row.get("weighted_sentiment", 0.0)),
                "return_lag_1": float(last_row.get("return_lag_1", 0.0)),
                "mention_count": float(last_row.get("mention_count", 0.0)),
            },
        )
