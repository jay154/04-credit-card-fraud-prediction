"""Feature engineering for the credit card fraud model.

Both ``notebooks/03_modeling.ipynb`` and ``app.py`` import :func:`build_features`
so the training data and the live app are built exactly the same way.
"""
from __future__ import annotations

import pandas as pd

# Columns that are blank on every row in this dataset.
DEAD_COLUMNS = [
    "echoBuffer",
    "merchantCity",
    "merchantState",
    "merchantZip",
    "posOnPremises",
    "recurringAuthInd",
]

# Identifier-like columns that must not be fed to the model as raw numbers.
ID_COLUMNS = ["accountNumber", "customerId", "cardLast4Digits", "cardCVV", "enteredCVV"]

# Categorical fields that can contain an empty string; "" is replaced with
# "missing" so it becomes its own category rather than a silent gap.
CATEGORICAL_FEATURES = [
    "transactionType",
    "merchantCategoryCode",
    "posEntryMode",
    "posConditionCode",
    "acqCountry",
    "merchantCountryCode",
]

NUMERIC_FEATURES = [
    "transactionAmount",
    "creditLimit",
    "availableMoney",
    "currentBalance",
    "hour",
    "dayofweek",
    "month",
    "accountAgeDays",
    "daysSinceAddressChange",
    "amountToCreditLimit",
    "availableRatio",
    "balanceRatio",
]

BOOLEAN_FEATURES = ["cardPresent", "expirationDateKeyInMatch", "cvvMismatch"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

TARGET = "isFraud"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction rows into the model's feature matrix.

    Returns a DataFrame with exactly the columns in :data:`FEATURE_COLUMNS`,
    in that order. The input is not modified.
    """
    out = df.copy()

    ts = pd.to_datetime(out["transactionDateTime"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month

    account_open = pd.to_datetime(out["accountOpenDate"], errors="coerce")
    address_change = pd.to_datetime(out["dateOfLastAddressChange"], errors="coerce")
    out["accountAgeDays"] = (ts - account_open).dt.days
    out["daysSinceAddressChange"] = (ts - address_change).dt.days

    # creditLimit has no zeros in this data (minimum is 250), so these ratios are
    # always defined.
    out["amountToCreditLimit"] = out["transactionAmount"] / out["creditLimit"]
    out["availableRatio"] = out["availableMoney"] / out["creditLimit"]
    out["balanceRatio"] = out["currentBalance"] / out["creditLimit"]

    out["cvvMismatch"] = (out["cardCVV"] != out["enteredCVV"]).astype(int)

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("string").replace("", pd.NA).fillna("missing")

    # 0/1 integers rather than bool, so every non-categorical feature is numeric
    # for any downstream estimator.
    for col in ("cardPresent", "expirationDateKeyInMatch"):
        out[col] = out[col].astype(bool).astype("int8")

    return out[FEATURE_COLUMNS].reset_index(drop=True)
