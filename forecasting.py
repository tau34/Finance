"""Small, leakage-aware price forecasting pipeline for OHLCV CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = {"30m": pd.Timedelta(minutes=30), "1h": pd.Timedelta(hours=1), "3h": pd.Timedelta(hours=3)}
FEATURE_COLUMNS = [
    "return_1", "return_3", "return_6", "return_12", "range_pct",
    "volume_change", "sma_ratio_6", "sma_ratio_24", "ema_ratio_12",
    "rsi_14", "volatility_12", "atr_14", "hour", "weekday",
]


def read_ohlcv_csv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    """Read yfinance's three-row-header CSV and return a normalized OHLCV frame."""
    raw = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError(f"{path} is not a yfinance-style multi-header CSV")
    values = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        matches = [column for column in raw.columns if str(column[0]) == field]
        if not matches:
            raise ValueError(f"{path} does not contain {field}")
        values[field.lower()] = raw[matches[0]]
    return normalize_ohlcv(pd.DataFrame(values), symbol or Path(path).stem)


def normalize_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize an OHLCV DataFrame returned by yfinance."""
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["symbol"] = symbol
    return frame


def read_ohlcv_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Read an OHLCV DataFrame returned directly by yfinance."""
    values = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        if isinstance(raw.columns, pd.MultiIndex):
            matches = [column for column in raw.columns if str(column[0]) == field]
        else:
            matches = [field] if field in raw.columns else []
        if not matches:
            raise ValueError(f"Downloaded data for {symbol} does not contain {field}")
        values[field.lower()] = raw[matches[0]]
    return normalize_ohlcv(pd.DataFrame(values), symbol)


def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule.endswith("m") and not rule.endswith("min"):
        rule = f"{rule[:-1]}min"
    result = frame[["open", "high", "low", "close", "volume"]].resample(
        rule, label="right", closed="right"
    ).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return result.dropna(subset=["open", "high", "low", "close"])


def _interval_minutes(interval: str) -> int:
    if interval.endswith("min"):
        return int(interval[:-3])
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60
    if interval.endswith("w"):
        return int(interval[:-1]) * 7 * 24 * 60
    raise ValueError(f"Unsupported interval: {interval}")


def make_features(frame: pd.DataFrame, feature_rule: str = "1h", suffix: str = "") -> pd.DataFrame:
    """Create features using only information available at each feature timestamp."""
    bars = _resample(frame, feature_rule)
    close = bars["close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - previous_close).abs(),
         (bars["low"] - previous_close).abs()], axis=1
    ).max(axis=1)
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    features = pd.DataFrame(index=bars.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["return_6"] = close.pct_change(6)
    features["return_12"] = close.pct_change(12)
    features["range_pct"] = (bars["high"] - bars["low"]) / close
    features["volume_change"] = bars["volume"].pct_change()
    features["sma_ratio_6"] = close / close.rolling(6).mean() - 1
    features["sma_ratio_24"] = close / close.rolling(24).mean() - 1
    features["ema_ratio_12"] = close / close.ewm(span=12, adjust=False).mean() - 1
    features["rsi_14"] = 100 - 100 / (1 + rs)
    features["volatility_12"] = close.pct_change().rolling(12).std()
    features["atr_14"] = true_range.rolling(14).mean() / close
    features["hour"] = features.index.hour
    features["weekday"] = features.index.day_of_week
    features = features.replace([np.inf, -np.inf], np.nan)
    if suffix:
        features = features.add_suffix(suffix)
    return features


def make_multi_timeframe_features(
    frame: pd.DataFrame, feature_intervals: Iterable[str] = ("1h",)
) -> pd.DataFrame:
    """Combine features calculated from all configured timeframes."""
    intervals = list(dict.fromkeys(feature_intervals))
    if not intervals:
        raise ValueError("feature_intervals must not be empty")
    base_interval = min(intervals, key=_interval_minutes)
    features = make_features(frame, base_interval)
    for interval in intervals:
        if interval == base_interval:
            continue
        lower = make_features(frame, interval, f"_{interval}")
        features = features.join(lower.reindex(features.index, method="ffill"))
    return features


def make_dataset(
    frame: pd.DataFrame, symbol_code: int = 0,
    feature_intervals: Iterable[str] = ("1h",),
) -> dict[str, pd.DataFrame]:
    """Build one dataset per horizon; targets are future log returns."""
    features = make_multi_timeframe_features(frame, feature_intervals)
    base_interval = min(dict.fromkeys(feature_intervals), key=_interval_minutes)
    target_close = _resample(frame, base_interval)["close"]
    base_minutes = _interval_minutes(base_interval)
    result = {}
    for name, horizon in HORIZONS.items():
        if horizon.total_seconds() % (base_minutes * 60) != 0:
            raise ValueError(
                f"{name} cannot be represented by {base_interval} bars; "
                "use a finer feature interval"
            )
        steps = int(horizon.total_seconds() // (base_minutes * 60))
        future = target_close.shift(-steps).reindex(features.index).to_numpy()
        current = target_close.reindex(features.index).to_numpy()
        data = features.copy()
        data["target_return"] = np.log(future / current)
        data["symbol_code"] = symbol_code
        feature_columns = list(features.columns)
        data = data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=feature_columns + ["target_return"]
        )
        data["target_direction"] = np.select(
            [data["target_return"] <= -0.0005, data["target_return"] >= 0.0005],
            [0, 2], default=1,
        ).astype(int)
        result[name] = data
    return result


def _build_models(model_name: str, seed: int = 42):
    if model_name == "logistic":
        classifier = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
        regressor = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
        return classifier, regressor
    return (
        LGBMClassifier(n_estimators=150, learning_rate=0.04, num_leaves=15, verbosity=-1, random_state=seed),
        LGBMRegressor(n_estimators=150, learning_rate=0.04, num_leaves=15, verbosity=-1, random_state=seed),
    )


def _fit_direction(model, x: pd.DataFrame, y: pd.Series):
    if y.nunique() < 2:
        return DummyClassifier(strategy="most_frequent").fit(x, y)
    return model.fit(x, y)


def train_one(data: pd.DataFrame, model_name: str, output: Path, label: str) -> dict:
    split = max(1, int(len(data) * 0.8))
    if split >= len(data):
        raise ValueError(f"{label}: not enough rows for a train/test split")
    feature_columns = [
        column for column in data.columns
        if column not in ("target_return", "target_direction", "symbol_code")
    ]
    x = data[feature_columns]
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_return = data["target_return"]
    y_direction = data["target_direction"]
    direction_model, return_model = _build_models(model_name)
    direction_model = _fit_direction(direction_model, x_train, y_direction.iloc[:split])
    return_model.fit(x_train, y_return.iloc[:split])
    predicted_return = return_model.predict(x_test)
    predicted_direction = direction_model.predict(x_test)
    metrics = {
        "rows": len(data), "train_rows": len(x_train), "test_rows": len(x_test),
        "return_mae": float(mean_absolute_error(y_return.iloc[split:], predicted_return)),
        "return_rmse": float(np.sqrt(mean_squared_error(y_return.iloc[split:], predicted_return))),
        "direction_accuracy": float(accuracy_score(y_direction.iloc[split:], predicted_direction)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"direction": direction_model, "return": return_model, "features": feature_columns}, output)
    return metrics


def train(
    paths: Iterable[str], scope: str, model_name: str, output_dir: str,
    feature_intervals: Iterable[str] = ("1h",),
) -> dict:
    frames = [read_ohlcv_csv(path) for path in paths]
    symbols = {frame["symbol"].iloc[0]: index for index, frame in enumerate(frames)}
    datasets = {
        frame["symbol"].iloc[0]: make_dataset(
            frame, symbols[frame["symbol"].iloc[0]], feature_intervals
        )
        for frame in frames
    }
    feature_intervals = list(feature_intervals)
    report = {
        "scope": scope, "model": model_name,
        "feature_intervals": feature_intervals, "horizons": {},
    }
    for horizon in HORIZONS:
        report["horizons"][horizon] = {}
        if scope in ("common", "both"):
            combined = pd.concat([data[horizon] for data in datasets.values()]).sort_index()
            report["horizons"][horizon]["common"] = train_one(
                combined, model_name, Path(output_dir) / f"common_{horizon}.joblib", f"common {horizon}"
            )
        if scope in ("individual", "both"):
            report["horizons"][horizon]["individual"] = {}
            for symbol, data in datasets.items():
                report["horizons"][horizon]["individual"][symbol] = train_one(
                    data[horizon], model_name, Path(output_dir) / f"{symbol}_{horizon}.joblib",
                    f"{symbol} {horizon}",
                )
    return report


def confirm_model_overwrite(output_dir: str | Path, overwrite: bool = False) -> None:
    existing = sorted(Path(output_dir).glob("*.joblib"))
    if not existing:
        return
    if overwrite:
        return
    answer = input(
        f"Models already exist in {output_dir} ({len(existing)} files). "
        "Overwrite them? [y/N]: "
    ).strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError("Training cancelled; existing models were not overwritten")


def write_evaluation_csv(report: dict, output: str | Path) -> None:
    """Write a compact per-model accuracy summary."""
    rows = []
    for period, period_report in report["horizons"].items():
        common = period_report.get("common")
        if common is not None:
            rows.append({
                "ticker": "common",
                "period": period,
                "bars": common["test_rows"],
                "accuracy": common["direction_accuracy"],
            })
        for ticker, metrics in period_report.get("individual", {}).items():
            rows.append({
                "ticker": ticker,
                "period": period,
                "bars": metrics["test_rows"],
                "accuracy": metrics["direction_accuracy"],
            })
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["ticker", "period", "bars", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 30m/1h/3h OHLCV forecasts.")
    parser.add_argument("--data", nargs="+", required=True, help="yfinance CSV file(s)")
    parser.add_argument("--scope", choices=("common", "individual", "both"), default="both")
    parser.add_argument("--model", choices=("lightgbm", "logistic"), default="lightgbm")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--feature-intervals", nargs="+", default=["1h"])
    parser.add_argument("--report", default="evaluation.json")
    parser.add_argument("--evaluation-csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    confirm_model_overwrite(args.output_dir, args.overwrite)
    report = train(args.data, args.scope, args.model, args.output_dir, args.feature_intervals)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    evaluation_csv = args.evaluation_csv or str(Path(args.report).with_name("evaluation.csv"))
    write_evaluation_csv(report, evaluation_csv)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
