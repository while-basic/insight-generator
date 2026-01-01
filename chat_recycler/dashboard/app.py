from __future__ import annotations

from pathlib import Path
from typing import Optional

import os

from flask import Flask, render_template

from chat_recycler.dashboard.data import (
    fetch_clusters,
    fetch_conversations,
    fetch_events,
    fetch_messages,
    fetch_metrics,
    fetch_variables,
)


def create_app(output_dir: Optional[Path] = None) -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    base_dir = output_dir or Path(os.environ.get("CHAT_RECYCLER_OUT", "out"))

    def db_path() -> Path:
        return base_dir / "db.sqlite"

    @app.route("/")
    def index() -> str:
        summaries = fetch_conversations(db_path())
        metrics = fetch_metrics(db_path())
        return render_template("index.html", summaries=summaries, metrics=metrics)

    @app.route("/conversations")
    def conversations() -> str:
        summaries = fetch_conversations(db_path())
        return render_template("conversations.html", summaries=summaries)

    @app.route("/conversations/<conversation_id>")
    def conversation_detail(conversation_id: str) -> str:
        messages = fetch_messages(db_path(), conversation_id)
        events = fetch_events(db_path(), conversation_id)
        variables = fetch_variables(db_path(), conversation_id)
        return render_template(
            "conversation_detail.html",
            conversation_id=conversation_id,
            messages=messages,
            events=events,
            variables=variables,
        )

    @app.route("/clusters")
    def clusters() -> str:
        cluster_rows = fetch_clusters(db_path())
        return render_template("clusters.html", clusters=cluster_rows)

    @app.route("/metrics")
    def metrics() -> str:
        metrics_data = fetch_metrics(db_path())
        return render_template("metrics.html", metrics=metrics_data)

    return app
