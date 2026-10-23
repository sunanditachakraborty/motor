"""Shared input validation and feature preparation for PMSM temperature prediction."""
from __future__ import annotations
import re
import pandas as pd

TARGET = "pm"
# Features are restricted to measurements expected to be available at prediction time.
FEATURE_CANDIDATES = ["ambient", "coolant", "u_d", "u_q", "motor_speed", "i_d", "i_q"]


def normalize_column_name(name: object) -> str:
    """Normalize common column-name variations without changing their meaning."""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    aliases = {
        "motorspeed": "motor_speed", "motor_speed_rpm": "motor_speed",
        "speed": "motor_speed", "profileid": "profile_id",
        "ud": "u_d", "uq": "u_q", "id": "i_d", "iq": "i_q",
    }
    return aliases.get(s, s)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    # If normalization created duplicate names, fail explicitly rather than silently choosing one.
    duplicates = out.columns[out.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate columns after name normalization: {duplicates}")
    return out


def prepare_features(df: pd.DataFrame, *, training: bool = False):
    """Apply the same column normalization, feature selection and numeric cleanup at train/inference time.

    Returns (X, y_or_None, metadata). Missing required predictors are an error; missing feature
    values are imputed using medians learned from the current input. For production, the saved
    model pipeline itself performs fitted median imputation, so prediction data is not independently
    median-imputed before inference.
    """
    data = normalize_columns(df)
    missing = [c for c in FEATURE_CANDIDATES if c not in data.columns]
    if missing:
        raise ValueError(
            f"Input is missing required feature column(s): {missing}. "
            f"Expected predictors: {FEATURE_CANDIDATES}. Check the database schema / CSV headers."
        )
    X = data[FEATURE_CANDIDATES].apply(pd.to_numeric, errors="coerce")
    # Infinite values are invalid sensor readings; treat them as missing for the fitted imputer.
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    y = None
    if training:
        if TARGET not in data.columns:
            raise ValueError(f"Target column '{TARGET}' was not found in the source table.")
        y = pd.to_numeric(data[TARGET], errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        # Require at least one observed value per feature; imputer handles partial missingness.
        all_missing = X.columns[X.isna().all()].tolist()
        if all_missing:
            raise ValueError(f"These predictors contain no usable numeric values: {all_missing}")
    return X, y, {"feature_names": FEATURE_CANDIDATES, "rows_input": len(data)}
