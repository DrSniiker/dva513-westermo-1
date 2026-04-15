from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_COLUMNS = {"id"}
TARGET_COLUMNS = {"attack_cat", "label"}


@dataclass
class FeatureBundle:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    preprocessor: ColumnTransformer


def build_feature_bundle(x_train_df: pd.DataFrame) -> FeatureBundle:
    candidate_columns = [c for c in x_train_df.columns if c not in (LEAKAGE_COLUMNS | TARGET_COLUMNS)]
    x = x_train_df[candidate_columns].copy()

    categorical_columns = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [c for c in x.columns if c not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", StandardScaler(), numeric_columns),
        ]
    )

    return FeatureBundle(
        feature_columns=candidate_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        preprocessor=preprocessor,
    )
