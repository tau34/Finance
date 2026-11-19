"""Generate the latest forecast from models created by forecasting.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import yfinance as yf

from forecasting import HORIZONS, _interval_minutes, make_multi_timeframe_features, read_ohlcv_frame

PERIOD_BY_INTERVAL = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d", "90m": "60d", "1h": "730d", "1d": "max",
}


def parse_prediction_time(value: str | None) -> pd.Timedelta:
    while True:
        selected = (value if value is not None else input(
            "Enter prediction reference time (e.g., latest / 1d / 3h): "
        )).strip().lower()
        if selected == "latest":
            return pd.Timedelta(0)
        match = re.fullmatch(r"(\d+)(m|h|d|w)", selected)
        if match:
            return pd.to_timedelta(int(match.group(1)), unit={
                "m": "m", "h": "h", "d": "d", "w": "w"
            }[match.group(2)])
        if value is not None:
            raise ValueError("Prediction reference must be latest or an offset such as 1d")
        print("Invalid prediction reference. Try latest, 1d, 3h, or 30m.")


def unique_output_path(directory: str | Path, ticker: str, scope: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = directory / f"prediction_{ticker}_{scope}_{stamp}.json"
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base.stem}_{counter}{base.suffix}"
        counter += 1
    return candidate


def get_download_period(interval: str, offset: pd.Timedelta) -> str:
    base = PERIOD_BY_INTERVAL.get(interval)
    if base is None:
        raise ValueError(f"Unsupported yfinance interval: {interval}")
    days = max(offset.days + 4, 4)
    if interval in ("1m", "2m", "5m", "15m", "30m", "90m"):
        return f"{min(days, 60)}d"
    if interval in ("60m", "1h"):
        return f"{min(days, 730)}d"
    return base


def get_download_periods(
    interval: str, offset: pd.Timedelta,
) -> list[str]:
    """Return small-to-large periods so latest predictions avoid needless downloads."""
    initial = get_download_period(interval, offset)
    candidates = [initial]
    if interval in ("1m", "2m", "5m", "15m", "30m", "90m"):
        candidates.extend(["7d", "30d", "60d"])
    elif interval in ("60m", "1h"):
        candidates.extend(["7d", "30d", "365d", "730d"])
    else:
        candidates.append("max")
    return list(dict.fromkeys(candidates))


def download_with_retries(
    ticker: str, interval: str, period: str, attempts: int = 3,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            downloaded = yf.download(
                tickers=ticker, period=period, interval=interval,
                auto_adjust=False, progress=False, group_by="column",
            )
            if not downloaded.empty:
                return downloaded
            last_error = RuntimeError(f"yfinance returned no data for {ticker}")
        except Exception as error:
            last_error = error
        if attempt < attempts:
            wait_seconds = 2 ** (attempt - 1)
            print(
                f"yfinance download failed for {ticker} "
                f"(attempt {attempt}/{attempts}); retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f"Unable to download yfinance data for {ticker} after {attempts} attempts"
    ) from last_error


def required_source_bars(
    source_interval: str, feature_intervals: list[str], offset: pd.Timedelta,
) -> int:
    source_minutes = _interval_minutes(source_interval)
    warmup = max(_interval_minutes(interval) * 25 for interval in feature_intervals)
    offset_bars = int(offset.total_seconds() // (source_minutes * 60))
    return max(1, (warmup // source_minutes) + offset_bars + 5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next 30m/1h/3h return and direction.")
    parser.add_argument("--ticker", required=True, help="yfinance ticker")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--bars", type=int, default=10000)
    parser.add_argument(
        "--prediction-time", "--as-of",
        help="Prediction reference: latest or offset such as 1d, 3h, 30m",
    )
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--scope", choices=("common", "individual"), default="individual")
    parser.add_argument("--feature-intervals", nargs="+", default=["1h"])
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if args.bars <= 0:
        raise ValueError("--bars must be positive")
    offset = parse_prediction_time(args.prediction_time)
    minimum_bars = required_source_bars(args.interval, args.feature_intervals, offset)
    downloaded = pd.DataFrame()
    selected_period = None
    for period in get_download_periods(args.interval, offset):
        downloaded = download_with_retries(args.ticker, args.interval, period)
        selected_period = period
        if len(downloaded) >= minimum_bars:
            break
        print(
            f"Only {len(downloaded)} source bars were returned for period {period}; "
            f"{minimum_bars} are required. Trying a larger period."
        )
    if len(downloaded) < minimum_bars:
        raise RuntimeError(
            f"Insufficient yfinance history for {args.ticker}: "
            f"downloaded {len(downloaded)} source bars using up to {selected_period}, "
            f"but at least {minimum_bars} are required for "
            f"{args.feature_intervals}."
        )
    input_bars = max(args.bars, minimum_bars)
    frame = read_ohlcv_frame(downloaded.tail(input_bars), args.ticker)
    if offset > pd.Timedelta(0):
        target_time = frame.index.max() - offset
        frame = frame.loc[frame.index <= target_time]
        if frame.empty:
            raise ValueError(
                f"No data exists at or before {target_time.isoformat()}; "
                "increase --bars or use a shorter prediction offset"
            )
    features = make_multi_timeframe_features(frame, args.feature_intervals)
    symbol = frame["symbol"].iloc[0]
    forecasts = {}
    for horizon in HORIZONS:
        filename = f"common_{horizon}.joblib" if args.scope == "common" else f"{symbol}_{horizon}.joblib"
        model_path = Path(args.model_dir) / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        bundle = joblib.load(model_path)
        feature_columns = bundle["features"]
        missing = [column for column in feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Feature columns missing from input: {missing}")
        usable_features = features.dropna(subset=feature_columns)
        if offset > pd.Timedelta(0):
            usable_features = usable_features.loc[usable_features.index <= target_time]
        if usable_features.empty:
            raise ValueError(
                "No complete feature row is available for prediction. "
                f"Downloaded {len(frame)} source bars with interval {args.interval}; "
                f"required at least {minimum_bars} bars for feature intervals "
                f"{args.feature_intervals}. Increase --bars or check that the "
                "model and feature_intervals match."
            )
        latest = usable_features.iloc[[-1]][feature_columns]
        direction_value = int(bundle["direction"].predict(latest)[0])
        forecasts[horizon] = {
            "timestamp": usable_features.index[-1].isoformat(),
            "predicted_log_return": float(bundle["return"].predict(latest)[0]),
            "direction": {0: "Short", 1: "Neutral", 2: "Long"}[direction_value],
        }
    result = {
        "symbol": symbol, "scope": args.scope,
        "prediction_reference": args.prediction_time or "interactive",
        "forecasts": forecasts,
    }
    output_path = Path(args.output) if args.output else unique_output_path(
        args.output_dir, symbol, args.scope
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_results = []
    if args.append and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError(f"Prediction output is not an array: {output_path}")
        output_results = existing
    output_results.append(result)
    output_path.write_text(json.dumps(output_results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **result}, indent=2))


if __name__ == "__main__":
    main()
