from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from chat_recycler.extract.schemas import PredictionRow, PredictionTarget


def _predict_from_model(features: pd.DataFrame, model_entry: object) -> List[str]:
    model, encoder = model_entry
    preds = model.predict(features)
    return encoder.inverse_transform(preds)


def generate_predictions(
    features: pd.DataFrame,
    models: Dict[str, object],
) -> List[PredictionRow]:
    rows: List[PredictionRow] = []
    feature_values = features.drop(columns=["conversation_id"], errors="ignore")
    next_artifact_preds: Optional[List[str]] = None
    next_blocker_preds: Optional[List[str]] = None
    completion_scores: Optional[List[float]] = None
    if "next_artifact" in models:
        next_artifact_preds = list(_predict_from_model(feature_values, models["next_artifact"]))
    if "next_blocker" in models:
        next_blocker_preds = list(_predict_from_model(feature_values, models["next_blocker"]))
    if "completion" in models:
        model = models["completion"]
        completion_scores = [float(score[1]) for score in model.predict_proba(feature_values)]
    for _, row in features.iterrows():
        convo_id = row["conversation_id"]
        if "next_artifact" in models:
            value = next_artifact_preds[features.index.get_loc(_)]
            rows.append(
                PredictionRow(
                    conversation_id=convo_id,
                    target=PredictionTarget.next_artifact,
                    prediction=value,
                    model="logreg",
                    features_used=list(feature_values.columns),
                    confidence=0.6,
                )
            )
        else:
            rows.append(
                PredictionRow(
                    conversation_id=convo_id,
                    target=PredictionTarget.next_artifact,
                    prediction="untrained",
                    model="rules",
                    features_used=list(feature_values.columns),
                    confidence=0.2,
                )
            )
        if "next_blocker" in models:
            value = next_blocker_preds[features.index.get_loc(_)]
            rows.append(
                PredictionRow(
                    conversation_id=convo_id,
                    target=PredictionTarget.next_blocker,
                    prediction=value,
                    model="logreg",
                    features_used=list(feature_values.columns),
                    confidence=0.6,
                )
            )
        else:
            rows.append(
                PredictionRow(
                    conversation_id=convo_id,
                    target=PredictionTarget.next_blocker,
                    prediction="untrained",
                    model="rules",
                    features_used=list(feature_values.columns),
                    confidence=0.2,
                )
            )
        if "completion" in models:
            score = completion_scores[features.index.get_loc(_)]
            rows.append(
                PredictionRow(
                    conversation_id=convo_id,
                    target=PredictionTarget.completion_likelihood,
                    prediction={"likelihood": score},
                    model="logreg",
                    features_used=list(feature_values.columns),
                    confidence=0.7,
                )
            )
        else:
            rows.append(
                PredictionRow(
                    conversation_id=convo_id,
                    target=PredictionTarget.completion_likelihood,
                    prediction={"likelihood": 0.5, "note": "untrained"},
                    model="rules",
                    features_used=list(feature_values.columns),
                    confidence=0.2,
                )
            )
    return rows
