"""Run forecasting.py or predict.py from a model-set JSON."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_set import data_paths, load_config, prompt_config_path


def run(command: list[str]) -> None:
    print(">", " ".join(command))
    subprocess.run(command, check=True)


def prompt_action() -> str:
    while True:
        value = input("Enter action (train / predict / simulate): ").strip().lower()
        if value in ("train", "predict", "simulate"):
            return value
        print("Invalid action. Enter train, predict, or simulate.")


def prompt_prediction_time() -> str:
    while True:
        value = input("Enter prediction reference (e.g. latest / 1d / 3h / 30m): ").strip()
        if value.lower() == "latest" or re.fullmatch(r"\d+[mhdw]", value.lower()):
            return value
        print("Invalid prediction reference. Enter latest, 1d, 3h, or 30m.")


def prompt_prediction_symbols(symbols: list) -> list[str]:
    tickers = [s["ticker"] if isinstance(s, dict) else s for s in symbols]
    print("Supported symbols:")
    for symbol in symbols:
        if isinstance(symbol, dict):
            print("  {} ({})".format(symbol["ticker"], symbol["name"]))
        else:
            print("  {}".format(symbol))
    print("  all (all symbols)")
    while True:
        value = input("Enter prediction symbol: ").strip()
        if value.lower() == "all":
            return tickers
        if value in tickers:
            return [value]
        print("Invalid symbol. Enter one of the listed symbols or all.")


def prediction_output_path(directory: Path, label: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = directory / f"prediction_{label}_{stamp}.json"
    counter = 1
    while candidate.exists():
        candidate = directory / f"prediction_{label}_{stamp}_{counter}.json"
        counter += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model set from model_set.json.")
    parser.add_argument("action", nargs="?", choices=("train", "predict", "simulate"))
    parser.add_argument("--config", help="Model-set directory or path to model_set.json")
    parser.add_argument("--symbol", help="Prediction symbol or all")
    parser.add_argument("--prediction-time", "--as-of", help="latest or an offset such as 1d")
    parser.add_argument("--scope", choices=("common", "individual", "both"), help="Model scope")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing trained models")
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    args = parser.parse_args()
    action = args.action or prompt_action()
    while True:
        try:
            config, set_dir = load_config(args.config or prompt_config_path())
            break
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            print(f"Invalid model-set: {error}")
            args.config = prompt_config_path()
    project_dir = Path(__file__).resolve().parent
    models_dir = set_dir / "models"
    if action == "train":
        paths = data_paths(config, set_dir)
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing CSV files: " + ", ".join(map(str, missing)))
        models_dir.mkdir(parents=True, exist_ok=True)
        selected_scope = args.scope or config["scope"]
        command = [
            sys.executable, str(project_dir / "forecasting.py"),
            "--data", *map(str, paths), "--scope", selected_scope,
            "--model", config["model"], "--output-dir", str(models_dir),
            "--feature-intervals", *config["data"]["feature_intervals"],
            "--report", str(set_dir / "evaluation.json"),
            "--evaluation-csv", str(set_dir / "evaluation.csv"),
        ]
        if args.overwrite:
            command.append("--overwrite")
        run(command)
        return
    if action == "simulate":
        output = set_dir / "equity.csv"
        scope = args.scope or config["scope"]
        run([
            sys.executable, str(project_dir / "simulation.py"),
            "--model-set", str(set_dir),
            "--scope", scope,
            "--fee-rate", str(args.fee_rate),
            "--slippage-rate", str(args.slippage_rate),
            "--output", str(output),
        ])
        return
    prediction_time = args.prediction_time or prompt_prediction_time()
    if prediction_time.lower() != "latest" and not re.fullmatch(r"\d+[mhdw]", prediction_time.lower()):
        raise ValueError("Invalid --prediction-time. Use latest or an offset such as 1d")
    if args.symbol:
        tickers = [s["ticker"] if isinstance(s, dict) else s for s in config["symbols"]]
        if args.symbol.lower() == "all":
            prediction_symbols = tickers
        elif args.symbol in tickers:
            prediction_symbols = [args.symbol]
        else:
            raise ValueError(f"Invalid --symbol. Choose one of {tickers} or all")
    else:
        prediction_symbols = prompt_prediction_symbols(config["symbols"])
    selected_scope = args.scope or config["scope"]
    scopes = [selected_scope] if selected_scope != "both" else ["common", "individual"]
    output_path = prediction_output_path(
        set_dir / "predictions",
        "all" if len(prediction_symbols) > 1 else prediction_symbols[0],
    )
    for scope in scopes:
        for prediction_symbol in prediction_symbols:
            run([
                sys.executable, str(project_dir / "predict.py"),
                "--ticker", prediction_symbol, "--interval", config["data"]["interval"],
                "--bars", str(config["data"]["bars"]),
                "--prediction-time", prediction_time,
                "--model-dir", str(models_dir), "--scope", scope,
                "--feature-intervals", *config["data"]["feature_intervals"],
                "--output", str(output_path), "--append",
            ])


if __name__ == "__main__":
    main()
