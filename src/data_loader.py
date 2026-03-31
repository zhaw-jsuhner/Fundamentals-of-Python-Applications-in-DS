from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def download_from_kaggle(dataset_slug: str, output_dir: Path) -> Path:
    """
    Download a Kaggle dataset zip via the official CLI.

    Requires local Kaggle setup (kaggle.json in the correct folder).
    Returns path to the downloaded zip file.
    """
    import subprocess

    ensure_directory(output_dir)
    command = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(output_dir)]
    subprocess.run(command, check=True)

    zip_name = f"{dataset_slug.split('/')[-1]}.zip"
    zip_path = output_dir / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Download expected at '{zip_path}', but file was not found. "
            "Please verify dataset slug and Kaggle credentials."
        )
    return zip_path


def extract_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    """Extract a zip file and return extracted CSV files."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    return sorted(output_dir.glob("*.csv"))


def pick_first_csv(raw_dir: Path) -> Path:
    """Pick the first CSV in data/raw if no explicit path is given."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in '{raw_dir}'. Download a dataset first or provide --csv-path."
        )
    return csv_files[0]


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_city_daytype_from_filename(path: Path) -> tuple[str, str]:
    """
    Expected filename format: <city>_weekdays.csv or <city>_weekends.csv
    Returns (city, day_type).
    """
    name = path.stem.lower()
    match = re.match(r"^(?P<city>.+)_(?P<day_type>weekdays|weekends)$", name)
    if not match:
        raise ValueError(
            f"Cannot parse city/day_type from filename '{path.name}'. "
            "Expected e.g. 'berlin_weekdays.csv'."
        )
    return match.group("city"), match.group("day_type")


def load_and_combine_raw_csvs(
    raw_dir: Path,
    pattern_weekdays: str = "*_weekdays.csv",
    pattern_weekends: str = "*_weekends.csv",
) -> pd.DataFrame:
    """Load all weekday/weekend CSVs from raw_dir and combine into one DataFrame."""
    csv_files = sorted(set(raw_dir.glob(pattern_weekdays)) | set(raw_dir.glob(pattern_weekends)))
    if not csv_files:
        raise FileNotFoundError(
            f"No matching CSVs found in '{raw_dir}'. Expected '*_weekdays.csv' / '*_weekends.csv'."
        )

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        city, day_type = parse_city_daytype_from_filename(csv_path)
        df = pd.read_csv(csv_path)
        df.insert(0, "city", city)
        df.insert(1, "day_type", day_type)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def run_first_eda(df: pd.DataFrame) -> None:
    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== INFO ===")
    print(df.info())

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== MISSING VALUES (TOP 15) ===")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    print("\n=== NUMERIC SUMMARY ===")
    print(df.describe())


def main() -> None:
    parser = argparse.ArgumentParser(description="Airbnb data ingestion + first EDA.")
    parser.add_argument("--dataset-slug", type=str, default="")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--combine-all", action="store_true")
    parser.add_argument("--output-path", type=str, default="data/processed/airbnb_combined.csv")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    ensure_directory(raw_dir)

    if args.dataset_slug:
        zip_path = download_from_kaggle(args.dataset_slug, raw_dir)
        csv_files = extract_zip(zip_path, raw_dir)
        if csv_files:
            print(f"Extracted CSV files: {[str(p) for p in csv_files]}")
        else:
            print("Dataset extracted, but no CSV found directly in data/raw.")

    if args.combine_all:
        output_path = Path(args.output_path)
        ensure_directory(output_path.parent)
        combined = load_and_combine_raw_csvs(raw_dir)
        combined.to_csv(output_path, index=False)
        print(f"Saved combined dataset to: {output_path} (rows={len(combined)})")
        run_first_eda(combined)
        return

    csv_path = Path(args.csv_path) if args.csv_path else pick_first_csv(raw_dir)
    df = load_data(csv_path)
    print(f"Loaded file: {csv_path}")
    run_first_eda(df)


if __name__ == "__main__":
    main()
