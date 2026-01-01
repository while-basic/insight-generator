from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def train_multiclass(features: pd.DataFrame, target: pd.Series) -> Tuple[Optional[LogisticRegression], Optional[LabelEncoder]]:
    if target.empty or target.nunique() < 2:
        return None, None
    encoder = LabelEncoder()
    y = encoder.fit_transform(target)
    model = LogisticRegression(max_iter=500)
    model.fit(features, y)
    return model, encoder


def train_completion_model(features: pd.DataFrame, target: pd.Series) -> Optional[LogisticRegression]:
    if target.empty or target.nunique() < 2:
        return None
    model = LogisticRegression(max_iter=500)
    model.fit(features, target)
    return model


def train_models(
    features: pd.DataFrame,
    labels: Optional[pd.DataFrame],
) -> Dict[str, object]:
    models: Dict[str, object] = {}
    if labels is None:
        return models
    merged = features.merge(labels, on="conversation_id", how="left")
    for target in ["next_artifact", "next_blocker"]:
        model, encoder = train_multiclass(merged.drop(columns=["conversation_id"]), merged[target].fillna("unknown"))
        if model:
            models[target] = (model, encoder)
    if "completion" in merged:
        model = train_completion_model(merged.drop(columns=["conversation_id"]), merged["completion"].fillna(0))
        if model:
            models["completion"] = model
    return models
