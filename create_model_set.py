"""Create a model-set directory and its default JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_set import validate_intervals


DEFAULT_SYMBOLS = [
    {"ticker": "AAPL", "name": "Apple", "market": "us_stock"},
    {"ticker": "MSFT", "name": "Microsoft", "market": "us_stock"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a model-set directory.")
    parser.add_argument("--name", default="default", help="Directory name under model_sets")
    parser.add_argument("--root", default="model_sets", help="Model-set root directory")
    parser.add_argument("--scope", choices=("common", "individual", "both"), default="both")
    parser.add_argument("--interval", default="5m", help="yfinance interval, e.g. 5m or 1h")
    parser.add_argument(
        "--feature-intervals", nargs="+", default=["1h"],
        help="Feature timeframes, e.g. 1h or 1h 10m",
    )
    parser.add_argument("--bars", type=int, default=10000, help="Number of latest bars to download")
    parser.add_argument("--symbols", nargs="+", default=None, help="Ticker symbols")
    args = parser.parse_args()
    if args.bars <= 0:
        parser.error("--bars must be positive")
    try:
        feature_intervals = validate_intervals(args.feature_intervals, "--feature-intervals")
    except ValueError as error:
        parser.error(str(error))
    symbols = [{"ticker": ticker, "name": ticker, "market": "unknown"}
               for ticker in (args.symbols or [item["ticker"] for item in DEFAULT_SYMBOLS])]
    set_dir = Path(args.root) / args.name
    (set_dir / "models").mkdir(parents=True, exist_ok=True)
    (set_dir / "data").mkdir(parents=True, exist_ok=True)
    config = {
        "name": args.name,
        "scope": args.scope,
        "model": "lightgbm",
        "data": {
            "interval": args.interval, "bars": args.bars, "directory": "data",
            "feature_intervals": feature_intervals,
        },
        "symbols": symbols,
    }
    config_path = set_dir / "model_set.json"
    if config_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {config_path}")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(config_path)


if __name__ == "__main__":
    main()
