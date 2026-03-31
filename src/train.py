from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def train_placeholder(df: pd.DataFrame, target: str) -> None:
    """Lightweight placeholder for upcoming model training."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    print(f"Training placeholder ready. Rows={len(df)}, target='{target}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regression models (placeholder).")
    parser.add_argument("--input-path", type=str, default="data/processed/airbnb_combined.csv")
    parser.add_argument("--target", type=str, default="realSum")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input_path))
    train_placeholder(df, args.target)


if __name__ == "__main__":
    main()
