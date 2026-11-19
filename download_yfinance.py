"""Download configured OHLCV CSV files from yfinance."""

from __future__ import annotations

import argparse
import json
import time
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
    if args.config:
        while True:
            try:
                config, set_dir = load_config(args.config)
                break
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
                print(f"Invalid --config value: {error}")
                args.config = prompt_config_path()
    else:
        config, set_dir = load_config(prompt_config_path())
    if args.start is not None:
        while True:
            try:
                start = validate_start(args.start)
                break
            except ValueError as error:
                print(f"Invalid --start value: {error}")
                args.start = prompt_start()
    else:
        start = prompt_start()
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
        downloaded = None
        last_error = None
        for attempt in range(1, 4):
            try:
                downloaded = yf.download(**options)
                if downloaded is not None and not downloaded.empty:
                    break
                last_error = RuntimeError(f"No data returned for {ticker}")
            except Exception as error:
                last_error = error
            if attempt < 3:
                wait_seconds = 2 ** (attempt - 1)
                print(f"{ticker}: download failed; retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
        if downloaded is None or downloaded.empty:
            raise RuntimeError(
                f"Unable to download yfinance data for {ticker} after 3 attempts"
            ) from last_error
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded.tail(bars).to_csv(destination)
        print(f"{ticker}: {len(downloaded.tail(bars))} rows -> {destination}")


if __name__ == "__main__":
    main()
