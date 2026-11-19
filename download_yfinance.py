"""Download configured OHLCV CSV files from yfinance."""

from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf

from model_set import data_paths, load_config, prompt_config_path, symbol_ticker


PERIOD_BY_INTERVAL = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d", "90m": "60d", "1h": "730d", "1d": "max",
}


def validate_start(value: str) -> str:
    from datetime import datetime
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("開始日時はYYYY-MM-DDまたはISO形式で指定してください") from error
    return value


def prompt_start() -> str:
    while True:
        value = input(
            "データ開始日時を入力してください（例: 2026-04-01、空欄で自動期間）: "
        ).strip()
        if not value:
            return ""
        try:
            return validate_start(value)
        except ValueError as error:
            print(f"入力が無効です: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download data for a model set.")
    parser.add_argument("--config", help="Model-set directory or path to model_set.json")
    parser.add_argument("--start", help="Data start date/time, e.g. 2026-04-01")
    args = parser.parse_args()
    config, set_dir = load_config(args.config or prompt_config_path())
    start = args.start if args.start is not None else prompt_start()
    interval = config["data"]["interval"]
    bars = int(config["data"]["bars"])
    if bars <= 0:
        raise ValueError("data.bars must be positive")
    period = PERIOD_BY_INTERVAL.get(interval)
    if period is None:
        raise ValueError(f"Unsupported yfinance interval for automatic period: {interval}")
    for symbol, destination in zip(config["symbols"], data_paths(config, set_dir)):
        ticker = symbol_ticker(symbol)
        options = {
            "tickers": ticker, "interval": interval,
            "auto_adjust": False, "progress": False, "group_by": "column",
        }
        if start:
            options["start"] = validate_start(start)
        else:
            options["period"] = period
        downloaded = yf.download(**options)
        if downloaded.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded.tail(bars).to_csv(destination)
        print(f"{ticker}: {len(downloaded.tail(bars))} rows -> {destination}")


if __name__ == "__main__":
    main()
