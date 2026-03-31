from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def evaluate_placeholder(df: pd.DataFrame, target: str) -> None:
    """Lightweight placeholder for upcoming model evaluation."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    print(f"Evaluation placeholder ready. Rows={len(df)}, target='{target}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models (placeholder).")
    parser.add_argument("--input-path", type=str, default="data/processed/airbnb_combined.csv")
    parser.add_argument("--target", type=str, default="realSum")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input_path))
    evaluate_placeholder(df, args.target)


if __name__ == "__main__":
    main()
