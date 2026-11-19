"""Model-set configuration helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


INTERVAL_PATTERN = re.compile(r"^[1-9]\d*(?:m|min|h|d|w)$")


def validate_intervals(intervals: object, field_name: str) -> list[str]:
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(f"{field_name} must be a non-empty list")
    normalized = list(dict.fromkeys(str(interval) for interval in intervals))
    invalid = [interval for interval in normalized if not INTERVAL_PATTERN.fullmatch(interval)]
    if invalid:
        raise ValueError(
            f"{field_name} contains invalid intervals {invalid}; "
            "use values such as 5m, 10m, 30m, 1h, or 1d"
        )
    return normalized


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    if config_path.is_dir():
        config_path /= "model_set.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model-set JSON not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("scope", "model", "symbols", "data"):
        if key not in config:
            raise ValueError(f"Model-set JSON is missing '{key}': {config_path}")
    if config["scope"] not in ("common", "individual", "both"):
        raise ValueError("scope must be common, individual, or both")
    if not isinstance(config["symbols"], list) or not config["symbols"]:
        raise ValueError("symbols must be a non-empty list")
    if not isinstance(config["data"], dict) or not config["data"].get("interval"):
        raise ValueError("data.interval is required")
    validate_intervals([str(config["data"]["interval"])], "data.interval")
    config["data"]["feature_intervals"] = validate_intervals(
        config["data"].get("feature_intervals", ["1h"]), "data.feature_intervals"
    )
    return config, config_path.parent


def prompt_config_path() -> str:
    while True:
        value = input(
            "Enter model-set directory or path to model_set.json "
            "(e.g., model_sets\\default): "
        ).strip()
        if not value:
            print("Model-set path cannot be empty.")
            continue
        try:
            load_config(value)
            return value
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            print(f"Invalid model-set: {error}")


def symbol_ticker(symbol: str | dict[str, Any]) -> str:
    ticker = symbol if isinstance(symbol, str) else symbol.get("ticker")
    if not ticker:
        raise ValueError(f"Each symbol must have a ticker: {symbol!r}")
    return str(ticker)


def data_paths(config: dict[str, Any], set_dir: Path) -> list[Path]:
    data_dir = set_dir / config["data"].get("directory", "data")
    return [data_dir / f"{symbol_ticker(symbol)}.csv" for symbol in config["symbols"]]
