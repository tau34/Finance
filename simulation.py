"""Walk-forward trade simulation for trained model sets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib

from forecasting import (
    HORIZONS,
    _interval_minutes,
    _resample,
    make_multi_timeframe_features,
    read_ohlcv_csv,
)
from model_set import load_config, symbol_ticker


def closed_return(direction: str, entry: float, exit_price: float,
                  fee_rate: float, slippage_rate: float) -> tuple[float, float]:
    gross = exit_price / entry - 1 if direction == "Long" else entry / exit_price - 1
    cost = 2 * (fee_rate + slippage_rate)
    return gross, gross - cost


def simulate_one(
    ticker: str,
    scope: str,
    horizon: str,
    frame,
    feature_intervals: list[str],
    model_dir: Path,
    fee_rate: float,
    slippage_rate: float,
) -> list[dict]:
    base_interval = min(feature_intervals, key=_interval_minutes)
    bars = _resample(frame, base_interval)
    features = make_multi_timeframe_features(frame, feature_intervals)
    model_name = f"common_{horizon}.joblib" if scope == "common" else f"{ticker}_{horizon}.joblib"
    model_path = model_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    bundle = joblib.load(model_path)
    feature_columns = bundle["features"]

    horizon_minutes = int(HORIZONS[horizon].total_seconds() // 60)
    base_minutes = _interval_minutes(base_interval)
    if horizon_minutes % base_minutes:
        raise ValueError(f"{horizon} cannot be represented by {base_interval} bars")
    hold_bars = horizon_minutes // base_minutes
    open_positions: list[dict] = []
    trades: list[dict] = []
    equity = 1.0
    peak = 1.0

    timestamps = bars.index
    for position in range(len(timestamps) - hold_bars - 1):
        timestamp = timestamps[position]
        if timestamp not in features.index:
            continue
        usable = features.dropna(subset=feature_columns)
        usable = usable.loc[usable.index <= timestamp]
        if usable.empty:
            continue
        latest = usable.iloc[[-1]][feature_columns]
        direction_value = int(bundle["direction"].predict(latest)[0])
        direction = {0: "Short", 1: "Neutral", 2: "Long"}[direction_value]
        entry_index = position + 1

        if direction != "Neutral" and entry_index + hold_bars < len(timestamps):
            entry_price = float(bars.iloc[entry_index]["open"])
            allocation = equity / (len(open_positions) + 1)
            open_positions.append({
                "ticker": ticker,
                "scope": scope,
                "horizon": horizon,
                "direction": direction,
                "entry_index": entry_index,
                "exit_index": entry_index + hold_bars,
                "entry_price": entry_price,
                "allocation": allocation,
            })

        still_open = []
        for trade in open_positions:
            if trade["exit_index"] != position + 1:
                still_open.append(trade)
                continue
            exit_price = float(bars.iloc[trade["exit_index"]]["open"])
            gross, net = closed_return(
                trade["direction"], trade["entry_price"], exit_price,
                fee_rate, slippage_rate,
            )
            pnl = trade["allocation"] * net
            equity += pnl
            peak = max(peak, equity)
            trades.append({
                "timestamp": timestamps[trade["exit_index"]].isoformat(),
                "ticker": trade["ticker"],
                "scope": trade["scope"],
                "horizon": trade["horizon"],
                "direction": trade["direction"],
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "allocation": trade["allocation"],
                "gross_return": gross,
                "net_return": net,
                "pnl": pnl,
                "equity": equity,
                "cumulative_return": equity - 1,
                "drawdown": equity / peak - 1,
            })
        open_positions = still_open
    return trades


def write_trades(path: Path, trades: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "ticker", "scope", "horizon", "direction",
        "entry_price", "exit_price", "allocation", "gross_return",
        "net_return", "pnl", "equity", "cumulative_return", "drawdown",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate trained model trades on historical CSV data.")
    parser.add_argument("--model-set", "--config", required=True)
    parser.add_argument("--scope", choices=("common", "individual", "both"), default=None)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.fee_rate < 0 or args.slippage_rate < 0:
        raise ValueError("Fee and slippage rates must be non-negative")
    config, set_dir = load_config(args.model_set)
    scope = args.scope or config["scope"]
    scopes = [scope] if scope != "both" else ["common", "individual"]
    feature_intervals = config["data"]["feature_intervals"]
    model_dir = set_dir / "models"
    data_dir = set_dir / config["data"].get("directory", "data")
    output = Path(args.output) if args.output else set_dir / "equity.csv"

    all_trades = []
    for symbol in config["symbols"]:
        ticker = symbol_ticker(symbol)
        csv_path = data_dir / f"{ticker}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing simulation CSV: {csv_path}")
        frame = read_ohlcv_csv(csv_path, ticker)
        for selected_scope in scopes:
            for horizon in ("1h", "3h"):
                all_trades.extend(simulate_one(
                    ticker, selected_scope, horizon, frame, feature_intervals,
                    model_dir, args.fee_rate, args.slippage_rate,
                ))
    all_trades.sort(key=lambda row: row["timestamp"])
    write_trades(output, all_trades)
    print(f"Wrote {len(all_trades)} trades -> {output}")


if __name__ == "__main__":
    main()
