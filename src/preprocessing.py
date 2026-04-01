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


def clean_data(
    df: pd.DataFrame,
    *,
    price_column: str = "price",
    max_price: float = 500.0,
    min_non_null_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Prepare data for modeling:
    - handle missing values
    - remove outliers
    - encode categorical features
    - perform light feature selection
    """
    cleaned = df.copy()

    # 1) Missing values: drop columns with too many NaNs, then impute remaining.
    min_non_null = max(1, int(len(cleaned) * min_non_null_ratio))
    cleaned = cleaned.dropna(axis=1, thresh=min_non_null)

    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    categorical_cols = cleaned.select_dtypes(exclude=["number"]).columns

    if len(numeric_cols) > 0:
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())
    if len(categorical_cols) > 0:
        cleaned[categorical_cols] = cleaned[categorical_cols].fillna("unknown")

    # 2) Outlier handling: domain cap for price + IQR filter for all numeric columns.
    if price_column in cleaned.columns:
        cleaned = cleaned[cleaned[price_column] < max_price]

    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]

    # 3) Categorical encoding.
    cleaned = pd.get_dummies(cleaned, drop_first=True)

    # 4) Feature selection: remove constant columns.
    nunique = cleaned.nunique(dropna=False)
    cleaned = cleaned.loc[:, nunique > 1]

    return cleaned.reset_index(drop=True)

