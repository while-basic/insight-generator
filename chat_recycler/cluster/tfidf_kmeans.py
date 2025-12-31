from __future__ import annotations

from collections import Counter
from typing import Dict, List

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from chat_recycler.extract.schemas import ClusterMethod, ClusterRow, MessageRow


def cluster_conversations(messages: List[MessageRow]) -> List[ClusterRow]:
    convo_text: Dict[str, str] = {}
    for message in messages:
        convo_text.setdefault(message.conversation_id, "")
        convo_text[message.conversation_id] += f" {message.content}"

    conversation_ids = list(convo_text.keys())
    if not conversation_ids:
        return []

    texts = [convo_text[cid] for cid in conversation_ids]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    n_clusters = min(3, len(conversation_ids))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    terms = vectorizer.get_feature_names_out()
    clusters: List[ClusterRow] = []
    for idx, cid in enumerate(conversation_ids):
        label = int(labels[idx])
        centroid = model.cluster_centers_[label]
        top_indices = centroid.argsort()[-5:][::-1]
        keywords = ", ".join(terms[i] for i in top_indices)
        cluster_label = f"cluster-{label}"
        clusters.append(
            ClusterRow(
                cluster_id=f"{cluster_label}-{cid}",
                conversation_id=cid,
                cluster_label=cluster_label,
                cluster_summary=f"Top terms: {keywords}",
                centroid_repr=keywords,
                method=ClusterMethod.tfidf_kmeans,
                confidence=0.5,
            )
        )
    return clusters


def summarize_cluster_terms(clusters: List[ClusterRow]) -> Dict[str, Counter[str]]:
    summary: Dict[str, Counter[str]] = {}
    for cluster in clusters:
        summary.setdefault(cluster.cluster_label, Counter())
        for term in cluster.centroid_repr.split(", "):
            summary[cluster.cluster_label][term] += 1
    return summary
