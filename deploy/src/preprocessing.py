import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# -----------------------------
# Target Variable
# -----------------------------
TARGET = "fare"

# -----------------------------
# Feature Lists
# -----------------------------

# Numerical features (raw + engineered)
NUM_FEATURES = [
    "nsmiles",
    "log_distance",
    "passengers",
    "log_passengers",
    "large_ms",
    "lf_ms",
    "has_low_cost_carrier",
    "is_monopoly_route"
]

# Categorical features
CAT_FEATURES = [
    "city1",
    "city2",
    "carrier_lg",
    "carrier_low",
    "quarter",
    "route_type"
]

# -----------------------------
# Data Cleaning
# -----------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw airfare data by removing invalid and extreme values.
    """
    df = df.copy()

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # Remove invalid fares
    df = df[df[TARGET] > 0]

    # Remove extreme outliers (top 1%)
    df = df[df[TARGET] < df[TARGET].quantile(0.99)]

    return df


# -----------------------------
# Feature Engineering
# -----------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates engineered features based on EDA insights.
    """
    df = df.copy()

    # Log transformations to handle skewness
    df["log_distance"] = np.log1p(df["nsmiles"])
    df["log_passengers"] = np.log1p(df["passengers"])

    # Competition indicators
    df["has_low_cost_carrier"] = (df["lf_ms"] > 0).astype(int)
    df["is_monopoly_route"] = (df["large_ms"] > 0.9).astype(int)

    # Route length categorization
    df["route_type"] = pd.cut(
        df["nsmiles"],
        bins=[0, 500, 1000, 2000, 5000],
        labels=["short", "medium", "long", "very_long"]
    )

    return df


# -----------------------------
# Preprocessing Pipeline
# -----------------------------
def build_preprocessor() -> ColumnTransformer:
    """
    Builds preprocessing pipeline with imputation for missing values.
    """

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_FEATURES),
            ("cat", cat_pipeline, CAT_FEATURES)
        ]
    )

    return preprocessor
