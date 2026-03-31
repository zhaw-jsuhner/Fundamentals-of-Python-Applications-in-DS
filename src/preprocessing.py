from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_processed_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal preprocessing baseline:
    - drop duplicate rows
    - reset index
    """
    return df.drop_duplicates().reset_index(drop=True)

