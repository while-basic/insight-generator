from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"


class EventType(str, Enum):
    intent = "intent"
    constraint = "constraint"
    decision = "decision"
    blocker = "blocker"
    action_request = "action_request"
    outcome = "outcome"
    next_step = "next_step"
    metric = "metric"
    tool_use = "tool_use"


class ExtractorType(str, Enum):
    heuristic = "heuristic"
    llm = "llm"


class VariableType(str, Enum):
    string = "string"
    int = "int"
    float = "float"
    bool = "bool"
    enum = "enum"
    json = "json"


class ClusterMethod(str, Enum):
    tfidf_kmeans = "tfidf_kmeans"
    embeddings_hdbscan = "embeddings_hdbscan"


class PredictionTarget(str, Enum):
    next_artifact = "next_artifact"
    next_blocker = "next_blocker"
    completion_likelihood = "completion_likelihood"


class MessageRow(BaseModel):
    conversation_id: str
    message_id: str
    created_at: Optional[datetime]
    role: Role
    content: str
    content_hash: str
    tokens_est: int
    source_file: str
    order_index: int


class EventRow(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    message_id: Optional[str]
    created_at: Optional[datetime]
    event_type: EventType
    label: str
    event_text: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    extractor: ExtractorType
    span: Dict[str, Optional[int]] = Field(default_factory=dict)


class VariableRow(BaseModel):
    var_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    message_id: Optional[str]
    created_at: Optional[datetime]
    var_name: str
    var_type: VariableType
    var_value: Any
    var_source_text: str
    confidence: float
    rule_id: str
    extractor: ExtractorType


class ClusterRow(BaseModel):
    cluster_id: str
    conversation_id: str
    cluster_label: str
    cluster_summary: str
    centroid_repr: str
    method: ClusterMethod
    confidence: float


class PredictionRow(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    target: PredictionTarget
    prediction: Any
    model: str
    features_used: List[str]
    confidence: float
